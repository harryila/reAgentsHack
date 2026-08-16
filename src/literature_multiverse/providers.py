"""Explicit, budgeted language-model provider boundary.

Ordinary tests and fixture pipelines use :class:`FixtureProvider`; importing this
module never imports an SDK, reads a credential file, or opens a network connection.
The Anthropic implementation performs exactly one Messages API attempt per call,
archives the result, and accounts for cost before another call is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

MODEL_RATES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
}
ALLOWED_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


class ProviderError(RuntimeError):
    """Base class for an explicit provider-boundary failure."""


class LiveProviderDisabled(ProviderError):
    """Raised when code tries to spend credits without the live opt-in."""


class ProviderBudgetExceeded(ProviderError):
    """Raised before a request whose conservative ceiling exceeds the budget."""


class ProviderAttemptExists(ProviderError):
    """Raised when a one-shot request key already has an archived attempt."""


def load_live_environment(env_path: str | Path, *, live_enabled: bool) -> None:
    """Load an optional mode-0600 environment file only after explicit live opt-in."""

    if not live_enabled:
        raise LiveProviderDisabled("loading live credentials requires explicit live opt-in")
    path = Path(env_path)
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ProviderError("live environment path must be a regular non-symlink file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise ProviderError(f"live environment file must have mode 0600, found {mode:04o}")
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ProviderError("python-dotenv is required to load the live environment") from exc
    load_dotenv(path, override=False)


class ProviderProtocol(Protocol):
    def generate(
        self,
        *,
        operation: str,
        request_key: str,
        prompt: str,
        system: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
    ) -> ProviderResult: ...


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ProviderResult:
    provider: str
    model: str
    response_id: str
    stop_reason: str
    text: str
    parsed_json: Any | None
    usage: ProviderUsage
    estimated_cost_usd: float
    request_sha256: str
    archive_path: Path


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _safe_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in value
    )
    safe = safe.strip("-.")
    if not safe:
        raise ValueError("provider operation/request key must contain a safe character")
    return safe


def estimate_cost_usd(model: str, usage: ProviderUsage) -> float:
    """Estimate first-party cost, conservatively charging cache writes at 2x input."""

    try:
        input_rate, output_rate = MODEL_RATES_USD_PER_MTOK[model]
    except KeyError as exc:
        raise ProviderError(f"no pinned price for live model {model!r}") from exc
    billable_input = usage.input_tokens
    billable_input += 2.0 * usage.cache_creation_input_tokens
    billable_input += 0.1 * usage.cache_read_input_tokens
    return (billable_input * input_rate + usage.output_tokens * output_rate) / 1_000_000


class ProviderBudget:
    """Cost ledger derived only from immutable provider attempt archives."""

    def __init__(self, archive_dir: str | Path, max_usd: float) -> None:
        if not math.isfinite(max_usd) or max_usd <= 0:
            raise ValueError("max_usd must be a positive finite number")
        self.archive_dir = Path(archive_dir)
        self.max_usd = float(max_usd)

    def spent_usd(self) -> float:
        total = 0.0
        if not self.archive_dir.exists():
            return total
        for path in sorted(self.archive_dir.rglob("*.provider.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                total += float(payload.get("estimated_cost_usd", 0.0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ProviderError(f"invalid provider cost archive: {path}") from exc
        return total

    def require_room(self, conservative_request_ceiling_usd: float) -> None:
        projected = self.spent_usd() + conservative_request_ceiling_usd
        if projected > self.max_usd + 1e-12:
            raise ProviderBudgetExceeded(
                f"provider budget would exceed ${self.max_usd:.2f}: "
                f"projected conservative total ${projected:.4f}"
            )


class FixtureProvider:
    """Deterministic response mapping for fixtures and all default tests."""

    def __init__(self, responses: Mapping[tuple[str, str], str | Mapping[str, Any]]) -> None:
        self._responses = dict(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        *,
        operation: str,
        request_key: str,
        prompt: str,
        system: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
    ) -> ProviderResult:
        del system, output_schema
        key = (operation, request_key)
        if key not in self._responses:
            raise ProviderError(f"no deterministic fixture response for {key!r}")
        self.calls.append(key)
        configured = self._responses[key]
        if isinstance(configured, Mapping):
            parsed: Any | None = dict(configured)
            text = canonical_json_bytes(parsed).decode("utf-8")
        else:
            text = configured
            parsed = None
        request_sha = sha256_json(
            {"operation": operation, "request_key": request_key, "prompt": prompt}
        )
        return ProviderResult(
            provider="fixture",
            model="fixture-stub",
            response_id=f"fixture:{request_sha[:16]}",
            stop_reason="end_turn",
            text=text,
            parsed_json=parsed,
            usage=ProviderUsage(input_tokens=0, output_tokens=0),
            estimated_cost_usd=0.0,
            request_sha256=request_sha,
            archive_path=Path("<fixture-memory>"),
        )


class AnthropicProvider:
    """One-attempt Anthropic Messages boundary with strict live and budget gates."""

    def __init__(
        self,
        *,
        model: str,
        effort: str,
        max_tokens: int,
        archive_dir: str | Path,
        max_budget_usd: float,
        live_enabled: bool,
        global_budget_dir: str | Path | None = None,
        global_max_budget_usd: float | None = None,
        client: Any | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if model not in MODEL_RATES_USD_PER_MTOK:
            raise ProviderError(f"live model is not price-pinned: {model!r}")
        if effort not in ALLOWED_EFFORTS:
            raise ValueError(f"unsupported effort {effort!r}")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.model = model
        self.effort = effort
        self.max_tokens = int(max_tokens)
        self.archive_dir = Path(archive_dir)
        self.budget = ProviderBudget(self.archive_dir, max_budget_usd)
        if (global_budget_dir is None) != (global_max_budget_usd is None):
            raise ValueError(
                "global_budget_dir and global_max_budget_usd must be supplied together"
            )
        self.global_budget = (
            ProviderBudget(global_budget_dir, global_max_budget_usd)
            if global_budget_dir is not None and global_max_budget_usd is not None
            else None
        )
        self.live_enabled = live_enabled
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))

    def _client_or_create(self) -> Any:
        if self._client is None:
            try:
                import anthropic  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ProviderError("install the anthropic SDK for explicit live calls") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def _conservative_ceiling(self, *, prompt: str, system: str | None) -> float:
        # Four UTF-8 bytes per token is intentionally conservative for ordinary English.
        estimated_input = math.ceil(
            (len(prompt.encode("utf-8")) + len((system or "").encode("utf-8"))) / 4
        )
        return estimate_cost_usd(
            self.model,
            ProviderUsage(input_tokens=estimated_input, output_tokens=self.max_tokens),
        )

    def generate(
        self,
        *,
        operation: str,
        request_key: str,
        prompt: str,
        system: str | None = None,
        output_schema: Mapping[str, Any] | None = None,
    ) -> ProviderResult:
        if not self.live_enabled:
            raise LiveProviderDisabled("Anthropic calls require an explicit --live opt-in")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        operation_safe = _safe_component(operation)
        request_key_safe = _safe_component(request_key)
        archive_path = self.archive_dir / f"{operation_safe}.{request_key_safe}.provider.json"
        if archive_path.exists():
            raise ProviderAttemptExists(f"archived provider attempt already exists: {archive_path}")

        output_config: dict[str, Any] = {"effort": self.effort}
        if output_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": dict(output_schema),
            }
        request_payload = {
            "operation": operation,
            "request_key": request_key,
            "provider": "anthropic",
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "system": system,
            "prompt": prompt,
            "output_schema": output_schema,
        }
        request_sha = sha256_json(request_payload)
        conservative_ceiling = self._conservative_ceiling(prompt=prompt, system=system)
        self.budget.require_room(conservative_ceiling)
        if self.global_budget is not None:
            self.global_budget.require_room(conservative_ceiling)
        attempted_at = self._clock().astimezone(UTC).isoformat()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system is not None:
            kwargs["system"] = system

        try:
            response = self._client_or_create().messages.create(**kwargs)
            stop_reason = str(getattr(response, "stop_reason", ""))
            text_blocks = [
                str(block.text)
                for block in getattr(response, "content", ())
                if getattr(block, "type", None) == "text"
            ]
            text = "".join(text_blocks)
            raw_usage = getattr(response, "usage", None)
            usage = ProviderUsage(
                input_tokens=int(getattr(raw_usage, "input_tokens", 0)),
                output_tokens=int(getattr(raw_usage, "output_tokens", 0)),
                cache_creation_input_tokens=int(
                    getattr(raw_usage, "cache_creation_input_tokens", 0) or 0
                ),
                cache_read_input_tokens=int(
                    getattr(raw_usage, "cache_read_input_tokens", 0) or 0
                ),
            )
            cost = estimate_cost_usd(self.model, usage)
            parsed: Any | None = None
            parse_failure = False
            if output_schema is not None and text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parse_failure = True
            status = (
                "complete"
                if stop_reason == "end_turn" and text and not parse_failure
                else "failed"
            )
            if parse_failure:
                failure = "PROVIDER_INVALID_STRUCTURED_JSON"
            elif status != "complete":
                failure = "PROVIDER_NON_TERMINAL_RESPONSE"
            else:
                failure = None
            archive = {
                **request_payload,
                "request_sha256": request_sha,
                "attempted_at": attempted_at,
                "status": status,
                "response_id": str(getattr(response, "id", "")),
                "stop_reason": stop_reason,
                "response_text": text,
                "parsed_json": parsed,
                "usage": asdict(usage),
                "estimated_cost_usd": cost,
                "cost_basis": "reported_usage",
                "failure": failure,
            }
            _atomic_json(archive_path, archive)
            if status != "complete":
                raise ProviderError(
                    f"Anthropic response ended with {stop_reason!r}; attempt archived"
                )
            return ProviderResult(
                provider="anthropic",
                model=self.model,
                response_id=archive["response_id"],
                stop_reason=stop_reason,
                text=text,
                parsed_json=parsed,
                usage=usage,
                estimated_cost_usd=cost,
                request_sha256=request_sha,
                archive_path=archive_path,
            )
        except Exception as exc:
            if not archive_path.exists():
                _atomic_json(
                    archive_path,
                    {
                        **request_payload,
                        "request_sha256": request_sha,
                        "attempted_at": attempted_at,
                        "status": "failed",
                        "response_id": None,
                        "stop_reason": None,
                        "response_text": None,
                        "parsed_json": None,
                        "usage": asdict(ProviderUsage(0, 0)),
                        "estimated_cost_usd": conservative_ceiling,
                        "cost_basis": "preflight_ceiling_after_unknown_failure",
                        "failure": type(exc).__name__,
                    },
                )
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(
                "Anthropic attempt failed and was archived; no retry was made"
            ) from exc
