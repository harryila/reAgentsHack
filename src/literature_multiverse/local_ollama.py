"""Strict, hashable localhost Ollama generation boundary.

The client in this module is intentionally small.  It accepts a frozen generation
configuration and an explicit JSON Schema, verifies the locally installed model by
digest, and returns the exact response text plus Ollama telemetry.  It cannot be pointed
at a non-loopback host.  This makes the same boundary reusable by offline diagnostics
and by later local prompt-optimization experiments without pretending that a local model
call is a hosted-provider call.
"""

from __future__ import annotations

import http.client
import json
import math
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from literature_multiverse.lineage import hash_canonical

LOCAL_OLLAMA_CLIENT_VERSION = "strict-localhost-ollama-client-v1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class LocalOllamaError(RuntimeError):
    """The local server, installed model, or response violated the frozen contract."""


class OllamaGenerationConfig(BaseModel):
    """All generation settings that can affect an Ollama response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: str = "ollama-deterministic-json-generation-v1"
    model: str
    model_digest: str
    expected_ollama_version: str
    seed: int
    temperature: float = Field(ge=0.0)
    top_k: int = Field(ge=1)
    top_p: float = Field(gt=0.0, le=1.0)
    num_ctx: int = Field(ge=1024)
    num_predict: int = Field(ge=1)
    keep_alive: str

    @field_validator("model_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("model_digest must be lowercase sha256 hexadecimal")
        return value

    @field_validator("model", "expected_ollama_version", "keep_alive")
    @classmethod
    def validate_nonempty_safe_text(cls, value: str) -> str:
        if not value or any(character in value for character in "\r\n\0"):
            raise ValueError("Ollama configuration text must be nonempty single-line text")
        return value

    @field_validator("temperature", "top_p")
    @classmethod
    def validate_finite_float(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Ollama generation floats must be finite")
        return value

    @property
    def config_sha256(self) -> str:
        return hash_canonical(self.model_dump(mode="json"))

    def request_options(self) -> dict[str, Any]:
        return {
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "seed": self.seed,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


class OllamaIdentity(BaseModel):
    """Observed local runtime and exact installed-model identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity_version: str = "ollama-local-runtime-identity-v1"
    client_version: str = LOCAL_OLLAMA_CLIENT_VERSION
    ollama_version: str
    model: str
    model_digest: str
    parameter_size: str
    quantization_level: str
    model_format: str
    model_family: str

    @property
    def identity_sha256(self) -> str:
        return hash_canonical(self.model_dump(mode="json"))


class OllamaGenerationResult(BaseModel):
    """Exact model text and the bounded telemetry used by diagnostic receipts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_version: str = "ollama-local-generation-result-v1"
    model: str
    response_text: str
    done: bool
    done_reason: str | None = None
    total_duration_ns: int | None = Field(default=None, ge=0)
    load_duration_ns: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration_ns: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration_ns: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_finished_generation(self) -> OllamaGenerationResult:
        if not self.done:
            raise ValueError("non-streaming Ollama generation did not finish")
        return self


class OllamaClientProtocol(Protocol):
    """Reusable boundary for deterministic schema-constrained local generation."""

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        """Verify and return the local runtime/model identity."""

    def generate(
        self,
        *,
        prompt: str,
        output_schema: Mapping[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        """Generate one non-streaming response under the exact frozen contract."""


class LocalOllamaClient:
    """Minimal HTTP client restricted to an explicit loopback Ollama endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", *, timeout_seconds: float = 300):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in _LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            raise LocalOllamaError(
                "Ollama endpoint must be an unauthenticated loopback HTTP origin"
            )
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise LocalOllamaError("Ollama timeout must be positive and finite")
        self._host = parsed.hostname
        self._port = parsed.port or 11434
        self._timeout_seconds = timeout_seconds

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload, separators=(",", ":"))
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection = http.client.HTTPConnection(
            self._host,
            self._port,
            timeout=self._timeout_seconds,
        )
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
        except (OSError, TimeoutError, http.client.HTTPException) as exc:
            raise LocalOllamaError(f"local Ollama request failed: {type(exc).__name__}") from exc
        finally:
            connection.close()
        if response.status < 200 or response.status >= 300:
            raise LocalOllamaError(f"local Ollama returned HTTP {response.status}")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LocalOllamaError("local Ollama returned invalid JSON") from exc
        return decoded

    def inspect_identity(self, config: OllamaGenerationConfig) -> OllamaIdentity:
        version_payload = self._request("GET", "/api/version")
        tags_payload = self._request("GET", "/api/tags")
        if not isinstance(version_payload, Mapping) or not isinstance(
            version_payload.get("version"), str
        ):
            raise LocalOllamaError("local Ollama version response is invalid")
        version = version_payload["version"]
        if version != config.expected_ollama_version:
            raise LocalOllamaError(
                "Ollama version mismatch: expected "
                f"{config.expected_ollama_version}, observed {version}"
            )
        models = tags_payload.get("models") if isinstance(tags_payload, Mapping) else None
        if not isinstance(models, list):
            raise LocalOllamaError("local Ollama tag response is invalid")
        matches = [
            model
            for model in models
            if isinstance(model, Mapping)
            and model.get("name") == config.model
            and model.get("digest") == config.model_digest
        ]
        if len(matches) != 1:
            raise LocalOllamaError(
                "exact configured Ollama model name/digest is not installed once"
            )
        details = matches[0].get("details")
        if not isinstance(details, Mapping):
            raise LocalOllamaError("local Ollama model details are absent")
        required = ("parameter_size", "quantization_level", "format", "family")
        if any(not isinstance(details.get(field), str) or not details[field] for field in required):
            raise LocalOllamaError("local Ollama model details are incomplete")
        return OllamaIdentity(
            ollama_version=version,
            model=config.model,
            model_digest=config.model_digest,
            parameter_size=details["parameter_size"],
            quantization_level=details["quantization_level"],
            model_format=details["format"],
            model_family=details["family"],
        )

    def generate(
        self,
        *,
        prompt: str,
        output_schema: Mapping[str, Any],
        config: OllamaGenerationConfig,
    ) -> OllamaGenerationResult:
        if not prompt:
            raise LocalOllamaError("Ollama prompt must be nonempty")
        payload = {
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "format": dict(output_schema),
            "keep_alive": config.keep_alive,
            "options": config.request_options(),
        }
        response = self._request("POST", "/api/generate", payload)
        if not isinstance(response, Mapping):
            raise LocalOllamaError("local Ollama generation response is not an object")
        projected = {
            "model": response.get("model"),
            "response_text": response.get("response"),
            "done": response.get("done"),
            "done_reason": response.get("done_reason"),
            "total_duration_ns": response.get("total_duration"),
            "load_duration_ns": response.get("load_duration"),
            "prompt_eval_count": response.get("prompt_eval_count"),
            "prompt_eval_duration_ns": response.get("prompt_eval_duration"),
            "eval_count": response.get("eval_count"),
            "eval_duration_ns": response.get("eval_duration"),
        }
        try:
            result = OllamaGenerationResult.model_validate(projected)
        except ValueError as exc:
            raise LocalOllamaError("local Ollama generation telemetry is invalid") from exc
        if result.model != config.model:
            raise LocalOllamaError("local Ollama generation returned a different model name")
        return result
