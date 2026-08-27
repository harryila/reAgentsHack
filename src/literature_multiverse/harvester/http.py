"""Polite, bounded HTTP transport used by all public harvester adapters."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlsplit

import httpx

from .contracts import RetrievedPayload


class HarvestHttpError(RuntimeError):
    """A public retrieval failed after the configured bounded attempts."""


class UnsafeHarvestUrl(ValueError):
    """A URL is not suitable for a public, unauthenticated fetch."""


def validate_public_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeHarvestUrl("harvest_url_requires_http_or_https")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeHarvestUrl("harvest_url_forbids_embedded_credentials")
    hostname = parsed.hostname.casefold().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise UnsafeHarvestUrl("harvest_url_forbids_localhost")
    try:
        address = ipaddress.ip_address(hostname.strip("[]"))
    except ValueError:
        return url
    if not address.is_global:
        raise UnsafeHarvestUrl("harvest_url_forbids_non_public_ip")
    return url


def _retry_after_seconds(value: str | None, now: datetime) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return max(0.0, (when - now).total_seconds())


class PoliteHttpClient:
    """Unauthenticated GET client with host pacing, redirects, retries, and byte caps."""

    def __init__(
        self,
        *,
        user_agent: str = "literature-multiverse-harvester/0.1",
        contact_email: str | None = None,
        timeout_seconds: float = 20.0,
        max_attempts: int = 3,
        min_interval_seconds: float = 0.1,
        max_response_bytes: int = 50_000_000,
        max_redirects: int = 5,
        transport: httpx.BaseTransport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if timeout_seconds <= 0 or max_attempts < 1 or min_interval_seconds < 0:
            raise ValueError("invalid_http_budget")
        if max_response_bytes < 1 or max_redirects < 0:
            raise ValueError("invalid_http_response_limit")
        if "\n" in user_agent or "\r" in user_agent:
            raise ValueError("invalid_http_user_agent")
        if contact_email and any(character in contact_email for character in "\r\n"):
            raise ValueError("invalid_http_contact")
        agent = user_agent if not contact_email else f"{user_agent} (mailto:{contact_email})"
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": agent},
            follow_redirects=False,
            transport=transport,
            # Reproducible public harvesting must not silently inherit ALL_PROXY,
            # HTTP_PROXY, or HTTPS_PROXY.  In particular, an ambient SOCKS URL makes
            # httpx require the optional ``socksio`` package while constructing the
            # client, before a bounded request can even be attempted.
            trust_env=False,
        )
        self.max_attempts = max_attempts
        self.min_interval_seconds = min_interval_seconds
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects
        self._sleeper = sleeper
        self._monotonic = monotonic
        self._last_request_by_host: dict[str, float] = {}

    def __enter__(self) -> PoliteHttpClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _pace(self, url: str) -> None:
        host = urlsplit(url).hostname or ""
        previous = self._last_request_by_host.get(host)
        now = self._monotonic()
        if previous is not None:
            remaining = self.min_interval_seconds - (now - previous)
            if remaining > 0:
                self._sleeper(remaining)
        self._last_request_by_host[host] = self._monotonic()

    def _request_once(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None,
        headers: Mapping[str, str] | None,
    ) -> tuple[RetrievedPayload, str | None]:
        validate_public_url(url)
        self._pace(url)
        with self._client.stream("GET", url, params=params, headers=headers) as response:
            body_parts: list[bytes] = []
            observed = 0
            for chunk in response.iter_bytes():
                observed += len(chunk)
                if observed > self.max_response_bytes:
                    raise HarvestHttpError(
                        f"response_too_large:limit={self.max_response_bytes}:url={response.url}"
                    )
                body_parts.append(chunk)
            body = b"".join(body_parts)
            selected_headers = {
                key.casefold(): value
                for key, value in response.headers.items()
                if key.casefold()
                in {"content-type", "etag", "last-modified", "retry-after", "content-length"}
            }
            payload = RetrievedPayload(
                url=str(response.url),
                retrieved_at=datetime.now(UTC),
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
                body=body,
                response_headers=selected_headers,
            )
            location = response.headers.get("location")
            return payload, location

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RetrievedPayload:
        """Fetch one public resource, retrying only transient failures."""

        initial_url = validate_public_url(url)
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            current_url = initial_url
            current_params = params
            try:
                for redirect_number in range(self.max_redirects + 1):
                    payload, location = self._request_once(
                        current_url,
                        params=current_params,
                        headers=headers,
                    )
                    current_params = None
                    if payload.status_code in {301, 302, 303, 307, 308} and location:
                        if redirect_number == self.max_redirects:
                            raise HarvestHttpError(f"too_many_redirects:url={initial_url}")
                        current_url = validate_public_url(urljoin(payload.url, location))
                        continue
                    break
                else:  # pragma: no cover - loop always exits by return/break/raise
                    raise HarvestHttpError(f"too_many_redirects:url={initial_url}")
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                self._sleeper(min(30.0, 0.5 * (2 ** (attempt - 1))))
                continue

            if 200 <= payload.status_code < 300:
                return payload
            if payload.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                raise HarvestHttpError(f"http_status:{payload.status_code}:url={payload.url}")
            last_error = HarvestHttpError(
                f"transient_http_status:{payload.status_code}:url={payload.url}"
            )
            if attempt < self.max_attempts:
                retry_after = _retry_after_seconds(
                    payload.response_headers.get("retry-after"), datetime.now(UTC)
                )
                delay = retry_after if retry_after is not None else 0.5 * (2 ** (attempt - 1))
                self._sleeper(min(30.0, delay))

        raise HarvestHttpError(
            f"http_attempts_exhausted:attempts={self.max_attempts}:url={initial_url}"
        ) from last_error


__all__ = ["HarvestHttpError", "PoliteHttpClient", "UnsafeHarvestUrl", "validate_public_url"]
