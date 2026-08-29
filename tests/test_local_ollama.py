from __future__ import annotations

from typing import Any

import pytest

from literature_multiverse.local_ollama import (
    LocalOllamaClient,
    LocalOllamaError,
    OllamaGenerationConfig,
)


def _config() -> OllamaGenerationConfig:
    return OllamaGenerationConfig(
        model="fixture:1b",
        model_digest="a" * 64,
        expected_ollama_version="0.15.1",
        seed=7,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        num_ctx=4096,
        num_predict=128,
        keep_alive="5m",
    )


class _RecordingLocalClient(LocalOllamaClient):
    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        super().__init__("http://127.0.0.1:11434")
        self.responses = responses
        self.requests: list[tuple[str, str, Any]] = []

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        self.requests.append((method, path, payload))
        return self.responses[(method, path)]


def test_client_rejects_every_non_loopback_or_authenticated_origin() -> None:
    for origin in (
        "https://127.0.0.1:11434",
        "http://example.com:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434/api",
    ):
        with pytest.raises(LocalOllamaError, match="loopback"):
            LocalOllamaClient(origin)


def test_identity_and_generation_bind_every_deterministic_setting() -> None:
    config = _config()
    client = _RecordingLocalClient(
        {
            ("GET", "/api/version"): {"version": "0.15.1"},
            ("GET", "/api/tags"): {
                "models": [
                    {
                        "name": "fixture:1b",
                        "digest": "a" * 64,
                        "details": {
                            "parameter_size": "1.2B",
                            "quantization_level": "Q8_0",
                            "format": "gguf",
                            "family": "llama",
                        },
                    }
                ]
            },
            ("POST", "/api/generate"): {
                "model": "fixture:1b",
                "response": '{"value":"ok"}',
                "done": True,
                "done_reason": "stop",
                "total_duration": 10,
                "load_duration": 1,
                "prompt_eval_count": 5,
                "prompt_eval_duration": 2,
                "eval_count": 3,
                "eval_duration": 7,
            },
        }
    )
    identity = client.inspect_identity(config)
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    result = client.generate(prompt="input only", output_schema=schema, config=config)

    assert identity.model_digest == "a" * 64
    assert identity.ollama_version == "0.15.1"
    assert len(identity.identity_sha256) == 64
    assert len(config.config_sha256) == 64
    assert result.response_text == '{"value":"ok"}'
    request = client.requests[-1][2]
    assert request == {
        "model": "fixture:1b",
        "prompt": "input only",
        "stream": False,
        "format": schema,
        "keep_alive": "5m",
        "options": {
            "num_ctx": 4096,
            "num_predict": 128,
            "seed": 7,
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }


def test_identity_fails_closed_on_version_or_digest_mismatch() -> None:
    config = _config()
    client = _RecordingLocalClient(
        {
            ("GET", "/api/version"): {"version": "0.14.0"},
            ("GET", "/api/tags"): {"models": []},
        }
    )
    with pytest.raises(LocalOllamaError, match="version mismatch"):
        client.inspect_identity(config)

