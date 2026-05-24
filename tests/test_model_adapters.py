import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from miniadk import Message, ModelResult, ModelStreamEvent, Tool, ToolCall, tool
from miniadk.models.anthropic import AnthropicModel, _default_max_tokens
from miniadk.models.factory import model
from miniadk.models.openai import OpenAIModel


def test_openai_model_builds_chat_completion_payload():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    payload = model.build_payload(
        messages=[Message("system", "sys"), Message("user", "hi")],
        tools=[greet],
    )

    assert payload["model"] == "demo"
    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    assert payload["tools"][0]["function"]["name"] == "greet"
    assert payload["tools"][0]["function"]["parameters"]["required"] == ["name"]


def test_openai_model_preserves_flat_tool_schema_constraints():
    search = Tool(
        name="search",
        description="Search.",
        input_schema={
            "query": {
                "type": "string",
                "description": "Search query.",
                "minLength": 2,
                "maxLength": 20,
                "pattern": "^[a-z]+$",
                "required": True,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
            "mode": {
                "enum": ["files", "text"],
            },
        },
        func=lambda **kwargs: kwargs,
    )
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[search])
    params = payload["tools"][0]["function"]["parameters"]

    assert params == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
                "minLength": 2,
                "maxLength": 20,
                "pattern": "^[a-z]+$",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
            },
            "mode": {
                "enum": ["files", "text"],
            },
        },
        "additionalProperties": False,
        "required": ["query"],
    }


def test_openai_model_omits_internal_any_type_from_object_tool_schema():
    inspect = Tool(
        name="inspect",
        description="Inspect.",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "any"},
                "options": {
                    "type": "object",
                    "properties": {
                        "mode": {"type": "any", "enum": ["fast", "deep"]},
                    },
                },
            },
            "additionalProperties": False,
        },
        func=lambda **kwargs: kwargs,
    )
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[inspect])

    assert payload["tools"][0]["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "value": {},
            "options": {
                "type": "object",
                "properties": {
                    "mode": {"enum": ["fast", "deep"]},
                },
            },
        },
        "additionalProperties": False,
    }


def test_openai_model_builds_payload_with_generation_options():
    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        temperature=0.2,
        max_tokens=2048,
    )

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 2048


def test_openai_model_can_merge_extra_payload_options():
    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        temperature=0.2,
        opts={"top_p": 0.9, "temperature": 0.1},
    )

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert payload["top_p"] == 0.9
    assert payload["temperature"] == 0.1


def test_openai_model_reads_generation_options_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.3")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "4096")

    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 4096


def test_openai_model_omits_unset_generation_options(monkeypatch):
    monkeypatch.delenv("OPENAI_TEMPERATURE", raising=False)
    monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)

    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert "temperature" not in payload
    assert "max_tokens" not in payload


def test_openai_model_accepts_transport_options():
    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        timeout=12,
        retries=2,
        retry_delay=0.1,
    )

    assert model.http_client.timeout_seconds == 12
    assert model.http_client.retries == 2
    assert model.http_client.retry_delay == 0.1


def test_openai_model_reads_transport_options_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT", "15")
    monkeypatch.setenv("OPENAI_RETRIES", "3")
    monkeypatch.setenv("OPENAI_RETRY_DELAY", "0.2")

    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    assert model.http_client.timeout_seconds == 15
    assert model.http_client.retries == 3
    assert model.http_client.retry_delay == 0.2


def test_openai_model_reads_generic_transport_options_from_env(monkeypatch):
    for name in ["OPENAI_TIMEOUT", "OPENAI_RETRIES", "OPENAI_RETRY_DELAY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIADK_MODEL_TIMEOUT", "16")
    monkeypatch.setenv("MINIADK_MODEL_RETRIES", "4")
    monkeypatch.setenv("MINIADK_MODEL_RETRY_DELAY", "0.3")

    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    assert model.http_client.timeout_seconds == 16
    assert model.http_client.retries == 4
    assert model.http_client.retry_delay == 0.3


def test_openai_model_keeps_explicit_http_client():
    class FakeHttpClient:
        pass

    client = FakeHttpClient()
    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=client,
        timeout=12,
        retries=2,
        retry_delay=0.1,
    )

    assert model.http_client is client


def test_model_helper_prefers_anthropic_when_env_has_both_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example.test")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-demo")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "openai-demo")

    built = model()

    assert isinstance(built, AnthropicModel)
    assert built.api_key == "anthropic-key"
    assert built.base_url == "https://anthropic.example.test"
    assert built.model == "claude-demo"


def test_model_helper_accepts_provider_env_override(monkeypatch):
    monkeypatch.setenv("MINIADK_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "openai-demo")

    built = model()

    assert isinstance(built, OpenAIModel)
    assert built.api_key == "openai-key"
    assert built.base_url == "https://openai.example.test/v1"
    assert built.model == "openai-demo"


def test_model_helper_can_build_openai_compatible_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "openai-demo")

    built = model()

    assert isinstance(built, OpenAIModel)
    assert built.api_key == "openai-key"
    assert built.base_url == "https://openai.example.test/v1"
    assert built.model == "openai-demo"


def test_model_helper_can_use_generic_miniadk_openai_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in [
        "MINIADK_MODEL_PROVIDER",
        "ANTHROPIC_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIADK_MODEL_KEY", "generic-key")
    monkeypatch.setenv("MINIADK_MODEL_URL", "https://openai.generic.test/v1")
    monkeypatch.setenv("MINIADK_MODEL_NAME", "generic-model")
    monkeypatch.setenv("MINIADK_MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("MINIADK_MODEL_MAX_TOKENS", "3000")
    monkeypatch.setenv("MINIADK_MODEL_TIMEOUT", "20")
    monkeypatch.setenv("MINIADK_MODEL_RETRIES", "5")
    monkeypatch.setenv("MINIADK_MODEL_RETRY_DELAY", "0.4")

    built = model()

    assert isinstance(built, OpenAIModel)
    assert built.api_key == "generic-key"
    assert built.base_url == "https://openai.generic.test/v1"
    assert built.model == "generic-model"
    assert built.temperature == 0.2
    assert built.max_tokens == 3000
    assert built.http_client.timeout_seconds == 20
    assert built.http_client.retries == 5
    assert built.http_client.retry_delay == 0.4


def test_model_helper_can_use_generic_miniadk_anthropic_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in [
        "MINIADK_MODEL_PROVIDER",
        "ANTHROPIC_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "OPENAI_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIADK_MODEL_KEY", "generic-key")
    monkeypatch.setenv("MINIADK_MODEL_URL", "https://anthropic.generic.test")
    monkeypatch.setenv("MINIADK_MODEL_NAME", "generic-claude")

    built = model()

    assert isinstance(built, AnthropicModel)
    assert built.api_key == "generic-key"
    assert built.base_url == "https://anthropic.generic.test"
    assert built.model == "generic-claude"


def test_model_helper_loads_nearest_env_file_for_short_path(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "MINIADK_MODEL_PROVIDER=openai",
                "OPENAI_API_KEY=env-file-key",
                "OPENAI_BASE_URL=https://openai.env-file.test/v1",
                "OPENAI_MODEL=env-file-demo",
            ]
        ),
        encoding="utf-8",
    )
    nested = tmp_path / "app"
    nested.mkdir()
    monkeypatch.chdir(nested)
    for name in [
        "MINIADK_MODEL_PROVIDER",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)

    built = model()

    assert isinstance(built, OpenAIModel)
    assert built.api_key == "env-file-key"
    assert built.base_url == "https://openai.env-file.test/v1"
    assert built.model == "env-file-demo"


def test_model_helper_accepts_explicit_provider_and_overrides():
    built = model(
        "openai",
        name="demo",
        api_key="key",
        base_url="https://api.example.test/v1",
        temperature=0.4,
        max_tokens=123,
        timeout=9,
        retries=1,
        retry_delay=0.05,
        opts={"top_p": 0.8},
    )

    assert isinstance(built, OpenAIModel)
    assert built.model == "demo"
    assert built.api_key == "key"
    assert built.base_url == "https://api.example.test/v1"
    assert built.temperature == 0.4
    assert built.max_tokens == 123
    assert built.http_client.timeout_seconds == 9
    assert built.http_client.retries == 1
    assert built.http_client.retry_delay == 0.05
    assert built.opts == {"top_p": 0.8}


def test_model_helper_accepts_claude_alias():
    built = model(
        "claude",
        name="demo",
        api_key="key",
        base_url="https://api.example.test",
    )

    assert isinstance(built, AnthropicModel)
    assert built.model == "demo"


def test_model_helper_reports_unknown_provider():
    with pytest.raises(ValueError, match="Unknown model provider: local"):
        model("local")


def test_model_helper_reports_unknown_provider_env(monkeypatch):
    monkeypatch.setenv("MINIADK_MODEL_PROVIDER", "local")

    with pytest.raises(ValueError, match="Unknown model provider: local"):
        model()


def test_model_helper_requires_provider_or_key_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in [
        "MINIADK_MODEL_PROVIDER",
        "ANTHROPIC_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_KEY",
        "OPENAI_API_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match=r"model\(\) requires provider"):
        model()


def test_openai_model_builds_streaming_payload():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    payload = model.build_payload(
        messages=[Message("user", "hi")],
        tools=[],
        stream=True,
    )

    assert payload["stream"] is True


async def test_openai_model_streams_text_and_tool_calls():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            assert payload["stream"] is True
            yield {"choices": [{"delta": {"content": "hel"}}]}
            yield {"choices": [{"delta": {"content": "lo"}}]}
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": '{"name":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '"Ada"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    events = [event async for event in model.stream([Message("user", "hi")], [])]

    assert [event.delta for event in events[:2]] == ["hel", "lo"]
    assert events[2].tool_call.index == 0
    assert events[2].tool_call.id == "call_1"
    assert events[2].tool_call.name == "greet"
    assert events[2].tool_call.arguments == '{"name":'
    assert events[3].tool_call.index == 0
    assert events[3].tool_call.arguments == '"Ada"}'
    assert events[-1].result.message == "hello"
    assert events[-1].result.tool_calls == [
        ToolCall(id="call_1", name="greet", arguments={"name": "Ada"})
    ]


async def test_openai_model_stream_handles_repeated_full_tool_identity():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": '{"name":',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": '"Ada"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    events = [event async for event in model.stream([Message("user", "hi")], [])]

    assert events[-1].result.tool_calls == [
        ToolCall(id="call_1", name="greet", arguments={"name": "Ada"})
    ]


async def test_openai_model_stream_accepts_object_tool_arguments_from_compatible_gateways():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": {"name": "Ada"},
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    events = [event async for event in model.stream([Message("user", "hi")], [])]

    assert events[0].tool_call.arguments == '{"name": "Ada"}'
    assert events[-1].result.tool_calls == [
        ToolCall(id="call_1", name="greet", arguments={"name": "Ada"})
    ]


async def test_openai_model_stream_treats_missing_tool_arguments_as_empty_object():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "ping", "arguments": None},
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    events = [event async for event in model.stream([Message("user", "hi")], [])]

    assert events[0].tool_call.arguments is None
    assert events[-1].result.tool_calls == [
        ToolCall(id="call_1", name="ping", arguments={})
    ]


def test_openai_model_accepts_base_url_or_full_endpoint():
    base = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")
    full = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1/chat/completions",
        model="demo",
    )

    assert base.endpoint_url == "https://api.example.test/v1/chat/completions"
    assert full.endpoint_url == "https://api.example.test/v1/chat/completions"


def test_openai_model_defaults_to_deepseek_chat_for_deepseek_gateways(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    model = OpenAIModel(api_key="key", base_url="https://api.deepseek.com")

    assert model.model == "deepseek-chat"


def test_openai_model_accepts_base_url_env_alias(monkeypatch):
    monkeypatch.delenv("OPENAI_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.alias.test/v1")

    model = OpenAIModel(api_key="key", model="demo")

    assert model.base_url == "https://api.alias.test/v1"
    assert model.endpoint_url == "https://api.alias.test/v1/chat/completions"


def test_openai_model_accepts_generic_miniadk_env_aliases(monkeypatch):
    for name in [
        "OPENAI_KEY",
        "OPENAI_API_KEY",
        "OPENAI_URL",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_TEMPERATURE",
        "OPENAI_MAX_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIADK_MODEL_KEY", "generic-key")
    monkeypatch.setenv("MINIADK_MODEL_URL", "https://generic.openai.test/v1")
    monkeypatch.setenv("MINIADK_MODEL_NAME", "generic-model")
    monkeypatch.setenv("MINIADK_MODEL_TEMPERATURE", "0.7")
    monkeypatch.setenv("MINIADK_MODEL_MAX_TOKENS", "1234")

    model = OpenAIModel()

    assert model.api_key == "generic-key"
    assert model.base_url == "https://generic.openai.test/v1"
    assert model.model == "generic-model"
    assert model.temperature == 0.7
    assert model.max_tokens == 1234


def test_openai_model_parses_tool_calls():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    result = model.parse_response(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "greet",
                                    "arguments": '{"name": "Ada"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )

    assert result.tool_calls == [ToolCall(name="greet", arguments={"name": "Ada"}, id="call_1")]


def test_openai_model_accepts_object_tool_arguments_from_compatible_gateways():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    result = model.parse_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "greet",
                                    "arguments": {"name": "Ada"},
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert result.tool_calls == [
        ToolCall(name="greet", arguments={"name": "Ada"}, id="call_1")
    ]


def test_openai_model_treats_missing_tool_arguments_as_empty_object():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    result = model.parse_response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "ping", "arguments": None},
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert result.tool_calls == [
        ToolCall(name="ping", arguments={}, id="call_1")
    ]


def test_openai_model_reports_malformed_response_shape():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    with pytest.raises(RuntimeError, match="OpenAI response did not include a message"):
        model.parse_response({"choices": []})


def test_openai_model_reports_malformed_tool_arguments():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    with pytest.raises(RuntimeError, match="OpenAI tool call greet arguments are not valid JSON"):
        model.parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": "{not json",
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )


def test_openai_model_reports_missing_tool_function_shape():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    with pytest.raises(RuntimeError, match="OpenAI tool call missing function"):
        model.parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                }
                            ]
                        }
                    }
                ]
            }
        )


def test_openai_model_reports_missing_tool_function_name():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    with pytest.raises(RuntimeError, match="OpenAI tool call missing function name"):
        model.parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {"arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        )


def test_openai_model_requires_tool_arguments_object():
    model = OpenAIModel(api_key="key", base_url="https://api.example.test/v1", model="demo")

    with pytest.raises(RuntimeError, match="OpenAI tool call greet arguments must be a JSON object"):
        model.parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": '["Ada"]',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )


async def test_openai_model_reports_malformed_streamed_tool_arguments():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": "{not json",
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    with pytest.raises(RuntimeError, match="OpenAI tool call greet arguments are not valid JSON"):
        async for _ in model.stream([Message("user", "hi")], []):
            pass


async def test_openai_model_reports_invalid_streamed_tool_argument_type():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "greet",
                                        "arguments": ["Ada"],
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    with pytest.raises(
        RuntimeError,
        match="OpenAI streamed tool call arguments must be a JSON object or string",
    ):
        async for _ in model.stream([Message("user", "hi")], []):
            pass


async def test_openai_model_reports_malformed_stream_tool_call_shape():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": "not an object",
                                }
                            ]
                        }
                    }
                ]
            }

    model = OpenAIModel(
        api_key="key",
        base_url="https://api.example.test/v1",
        model="demo",
        http_client=FakeHttpClient(),
    )

    with pytest.raises(RuntimeError, match="OpenAI streamed tool call missing function"):
        async for _ in model.stream([Message("user", "hi")], []):
            pass


def test_anthropic_model_builds_messages_payload():
    @tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"hello {name}"

    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    payload = model.build_payload(
        messages=[Message("system", "sys"), Message("user", "hi")],
        tools=[greet],
    )

    assert payload["model"] == "demo"
    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["tools"][0]["name"] == "greet"
    assert payload["tools"][0]["input_schema"]["required"] == ["name"]


def test_anthropic_model_preserves_flat_tool_schema_constraints():
    search = Tool(
        name="search",
        description="Search.",
        input_schema={
            "query": {
                "type": "string",
                "minLength": 2,
                "required": True,
            },
            "args": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "object", "additionalProperties": {"type": "string"}},
                ],
            },
        },
        func=lambda **kwargs: kwargs,
    )
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[search])

    assert payload["tools"][0]["input_schema"]["properties"] == {
        "query": {"type": "string", "minLength": 2},
        "args": {
            "oneOf": [
                {"type": "string"},
                {"type": "object", "additionalProperties": {"type": "string"}},
            ],
        },
    }


def test_anthropic_model_builds_payload_with_generation_options():
    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        temperature=0.2,
        max_tokens=2048,
    )

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 2048


def test_anthropic_model_can_merge_extra_payload_options():
    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        temperature=0.2,
        opts={"top_p": 0.9, "temperature": 0.1},
    )

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert payload["top_p"] == 0.9
    assert payload["temperature"] == 0.1


def test_anthropic_model_reads_generation_options_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TEMPERATURE", "0.3")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "4096")

    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    payload = model.build_payload(messages=[Message("user", "hi")], tools=[])

    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 4096


def test_anthropic_model_accepts_transport_options():
    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        timeout=12,
        retries=2,
        retry_delay=0.1,
    )

    assert model.http_client.timeout_seconds == 12
    assert model.http_client.retries == 2
    assert model.http_client.retry_delay == 0.1


def test_anthropic_model_reads_transport_options_from_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_TIMEOUT", "15")
    monkeypatch.setenv("ANTHROPIC_RETRIES", "3")
    monkeypatch.setenv("ANTHROPIC_RETRY_DELAY", "0.2")

    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    assert model.http_client.timeout_seconds == 15
    assert model.http_client.retries == 3
    assert model.http_client.retry_delay == 0.2


def test_anthropic_model_reads_generic_transport_options_from_env(monkeypatch):
    for name in ["ANTHROPIC_TIMEOUT", "ANTHROPIC_RETRIES", "ANTHROPIC_RETRY_DELAY"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIADK_MODEL_TIMEOUT", "16")
    monkeypatch.setenv("MINIADK_MODEL_RETRIES", "4")
    monkeypatch.setenv("MINIADK_MODEL_RETRY_DELAY", "0.3")

    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    assert model.http_client.timeout_seconds == 16
    assert model.http_client.retries == 4
    assert model.http_client.retry_delay == 0.3


def test_anthropic_model_keeps_explicit_http_client():
    class FakeHttpClient:
        pass

    client = FakeHttpClient()
    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        http_client=client,
        timeout=12,
        retries=2,
        retry_delay=0.1,
    )

    assert model.http_client is client


def test_anthropic_model_builds_streaming_payload():
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    payload = model.build_payload(
        messages=[Message("user", "hi")],
        tools=[],
        stream=True,
    )

    assert payload["stream"] is True


async def test_anthropic_model_streams_text_and_tool_use_blocks():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            assert payload["stream"] is True
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            }
            yield {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "thinking", "thinking": ""},
            }
            yield {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "thinking_delta", "thinking": "checking"},
            }
            yield {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "greet",
                    "input": {},
                },
            }
            yield {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"name": "Ada"}'},
            }

    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        http_client=FakeHttpClient(),
    )

    events = [event async for event in model.stream([Message("user", "hi")], [])]

    assert events[0].delta == "hello"
    assert events[1].thinking == "checking"
    assert events[2].tool_call.index == 2
    assert events[2].tool_call.id == "toolu_1"
    assert events[2].tool_call.name == "greet"
    assert events[2].tool_call.arguments == '{"name": "Ada"}'
    assert events[-1].result.message == "hello"
    assert events[-1].result.tool_calls == [
        ToolCall(id="toolu_1", name="greet", arguments={"name": "Ada"})
    ]
    assert events[-1].result.content_blocks == [
        {"type": "text", "text": "hello"},
        {"type": "thinking", "thinking": "checking"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "greet",
            "input": {"name": "Ada"},
        },
    ]


async def test_anthropic_model_stream_uses_text_from_final_blocks():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": "hello from block"},
            }
            yield {"type": "content_block_stop", "index": 0}
            yield {"type": "message_stop"}

    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        http_client=FakeHttpClient(),
    )

    events = [event async for event in model.stream([Message("user", "hi")], [])]

    assert events == [
        ModelStreamEvent(
            result=ModelResult(
                message="hello from block",
                content_blocks=[{"type": "text", "text": "hello from block"}],
            )
        )
    ]


def test_anthropic_model_groups_multiple_tool_results_after_tool_use_message():
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    payload = model.build_payload(
        messages=[
            Message("system", "sys"),
            Message("user", "read project"),
            Message(
                "assistant",
                "I will read files.",
                tool_calls=[
                    ToolCall(
                        id="call_a",
                        name="read_file",
                        arguments={"path": "README.md"},
                    ),
                    ToolCall(
                        id="call_b",
                        name="read_file",
                        arguments={"path": "pyproject.toml"},
                    ),
                ],
            ),
            Message("tool", "readme text", name="read_file", tool_call_id="call_a"),
            Message("tool", "toml text", name="read_file", tool_call_id="call_b"),
        ],
        tools=[],
    )

    assert payload["messages"] == [
        {"role": "user", "content": "read project"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will read files."},
                {
                    "type": "tool_use",
                    "id": "call_a",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
                {
                    "type": "tool_use",
                    "id": "call_b",
                    "name": "read_file",
                    "input": {"path": "pyproject.toml"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_a",
                    "content": "readme text",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_b",
                    "content": "toml text",
                },
            ],
        },
    ]


def test_anthropic_model_preserves_thinking_blocks_for_followup_requests():
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")
    blocks = [
        {"type": "thinking", "thinking": "private reasoning", "signature": "sig"},
        {"type": "text", "text": "I will inspect."},
        {
            "type": "tool_use",
            "id": "call_a",
            "name": "read_file",
            "input": {"path": "README.md"},
        },
    ]

    result = model.parse_response({"content": blocks})

    assert result.message == "I will inspect."
    assert result.tool_calls == [
        ToolCall(id="call_a", name="read_file", arguments={"path": "README.md"})
    ]
    assert result.content_blocks == blocks

    payload = model.build_payload(
        messages=[
            Message("user", "read project"),
            Message(
                "assistant",
                result.message or "",
                tool_calls=result.tool_calls,
                content_blocks=result.content_blocks,
            ),
            Message("tool", "readme text", name="read_file", tool_call_id="call_a"),
        ],
        tools=[],
    )

    assert payload["messages"] == [
        {"role": "user", "content": "read project"},
        {"role": "assistant", "content": blocks},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_a",
                    "content": "readme text",
                }
            ],
        },
    ]


def test_anthropic_model_accepts_base_url_or_full_endpoint():
    base = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")
    full = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test/v1/messages",
        model="demo",
    )

    assert base.endpoint_url == "https://api.example.test/v1/messages"
    assert full.endpoint_url == "https://api.example.test/v1/messages"


def test_anthropic_model_uses_env_model_when_available(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "custom-claude")

    model = AnthropicModel(api_key="key", base_url="https://api.example.test")

    assert model.model == "custom-claude"


def test_anthropic_model_accepts_base_url_and_auth_token_env_aliases(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.alias.test")

    model = AnthropicModel(model="demo")

    assert model.api_key == "token"
    assert model.base_url == "https://anthropic.alias.test"
    assert model.endpoint_url == "https://anthropic.alias.test/v1/messages"


def test_anthropic_model_accepts_generic_miniadk_env_aliases(monkeypatch):
    for name in [
        "ANTHROPIC_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_URL",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_TEMPERATURE",
        "ANTHROPIC_MAX_TOKENS",
    ]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINIADK_MODEL_KEY", "generic-key")
    monkeypatch.setenv("MINIADK_MODEL_URL", "https://generic.anthropic.test")
    monkeypatch.setenv("MINIADK_MODEL_NAME", "generic-claude")
    monkeypatch.setenv("MINIADK_MODEL_TEMPERATURE", "0.7")
    monkeypatch.setenv("MINIADK_MODEL_MAX_TOKENS", "1234")

    model = AnthropicModel()

    assert model.api_key == "generic-key"
    assert model.base_url == "https://generic.anthropic.test"
    assert model.model == "generic-claude"
    assert model.temperature == 0.7
    assert model.max_tokens == 1234


def test_anthropic_model_default_model_is_current_haiku(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    model = AnthropicModel(api_key="key", base_url="https://api.example.test")

    assert model.model == "claude-4-5-haiku-latest"


def test_model_helper_passes_generation_options_to_anthropic():
    built = model(
        "anthropic",
        name="demo",
        api_key="key",
        base_url="https://api.example.test",
        temperature=0.4,
        max_tokens=123,
        opts={"top_p": 0.8},
    )

    assert isinstance(built, AnthropicModel)
    assert built.temperature == 0.4
    assert built.max_tokens == 123
    assert built.opts == {"top_p": 0.8}


def test_anthropic_model_parses_tool_use_blocks():
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    result = model.parse_response(
        {
            "content": [
                {"type": "text", "text": "I will use a tool."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "greet",
                    "input": {"name": "Ada"},
                },
            ]
        }
    )

    assert result.message == "I will use a tool."
    assert result.tool_calls == [ToolCall(name="greet", arguments={"name": "Ada"}, id="toolu_1")]


def test_anthropic_model_reports_malformed_tool_use_blocks():
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    with pytest.raises(RuntimeError, match="Anthropic tool_use block missing id or name"):
        model.parse_response({"content": [{"type": "tool_use", "id": "toolu_1"}]})


def test_anthropic_model_requires_tool_use_input_object():
    model = AnthropicModel(api_key="key", base_url="https://api.example.test", model="demo")

    with pytest.raises(RuntimeError, match="Anthropic tool_use block input must be an object"):
        model.parse_response(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "greet",
                        "input": ["Ada"],
                    }
                ]
            }
        )


async def test_anthropic_model_reports_malformed_streamed_tool_use_input():
    """Unrepairable JSON degrades gracefully — surface a sentinel input
    plus a repair note rather than aborting the whole turn."""

    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "greet",
                },
            }
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{not json"},
            }

    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        http_client=FakeHttpClient(),
    )

    final = None
    async for event in model.stream([Message("user", "hi")], []):
        if event.result is not None:
            final = event.result

    assert final is not None
    assert final.tool_calls and final.tool_calls[0].name == "greet"
    assert "_miniadk_invalid_input" in final.tool_calls[0].arguments
    # The block carries a human-readable repair note for the runtime / TUI.
    assert any(
        block.get("_partial_json_repair")
        for block in (final.content_blocks or [])
    )


async def test_anthropic_model_repairs_truncated_streamed_tool_use_input():
    """Truncated JSON (e.g. max_tokens cutoff) gets auto-repaired."""

    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "write_file",
                },
            }
            # Simulate a truncation mid-string — what max_tokens does in
            # practice. The "}\"" at the end is missing entirely.
            yield {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"path": "x.md", "content": "hello',
                },
            }

    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        http_client=FakeHttpClient(),
    )

    final = None
    async for event in model.stream([Message("user", "hi")], []):
        if event.result is not None:
            final = event.result

    assert final is not None
    call = final.tool_calls[0]
    assert call.arguments == {"path": "x.md", "content": "hello"}
    note = next(
        block.get("_partial_json_repair")
        for block in (final.content_blocks or [])
        if block.get("type") == "tool_use"
    )
    assert note and "auto-repaired" in note


async def test_anthropic_model_reports_streamed_tool_use_missing_identity():
    class FakeHttpClient:
        async def post_sse(self, url, payload, headers):
            yield {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "input": {},
                },
            }

    model = AnthropicModel(
        api_key="key",
        base_url="https://api.example.test",
        model="demo",
        http_client=FakeHttpClient(),
    )

    with pytest.raises(RuntimeError, match="Anthropic streamed tool_use block missing id or name"):
        async for _ in model.stream([Message("user", "hi")], []):
            pass


def test_model_adapters_hide_http_error_bodies():
    error = HTTPError(
        url="https://example.test",
        code=401,
        msg="Unauthorized",
        hdrs={},
        fp=BytesIO(b'{"error":"secret-token leaked in body"}'),
    )

    with pytest.raises(RuntimeError) as openai_error:
        raise OpenAIModel._http_error(error)

    assert str(openai_error.value) == "Model request failed with HTTP 401: Unauthorized"
    assert "secret-token" not in str(openai_error.value)

    error = HTTPError(
        url="https://example.test",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=BytesIO(b'{"error":"anthropic-secret leaked in body"}'),
    )

    with pytest.raises(RuntimeError) as anthropic_error:
        raise AnthropicModel._http_error(error)

    assert str(anthropic_error.value) == "Model request failed with HTTP 403: Forbidden"
    assert "anthropic-secret" not in str(anthropic_error.value)


# ─── max_tokens defaults table ───────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        # Claude 4.x
        ("claude-opus-4-7", 64_000),
        ("claude-opus-4-7-20260416", 64_000),
        ("claude-sonnet-4-6", 64_000),
        ("claude-haiku-4-5", 16_384),
        # GPT-5.x
        ("gpt-5", 128_000),
        ("gpt-5.1", 128_000),
        ("gpt-5.5", 128_000),
        ("gpt-5-pro", 128_000),
        # DeepSeek V4
        ("deepseek-v4-pro", 65_536),
        ("deepseek-v4-flash", 32_768),
        ("deepseek-v4", 32_768),
        # MiniMax
        ("minimax-2.7", 65_536),
        ("minimax-2.5", 65_536),
        # GLM
        ("glm-5", 32_768),
        ("glm-5.1", 32_768),
        # Kimi
        ("kimi-2.6", 65_536),
        ("kimi-2.5", 65_536),
        # Qwen
        ("qwen3-coder", 65_536),
        # Unknown → fallback
        ("totally-made-up-model-7b", 32_768),
        ("", 32_768),
    ],
)
def test_anthropic_default_max_tokens_table(model, expected):
    assert _default_max_tokens(model) == expected


def test_anthropic_model_uses_table_for_default_max_tokens(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)
    monkeypatch.delenv("MINIADK_MODEL_MAX_TOKENS", raising=False)

    sonnet = AnthropicModel(api_key="x", base_url="https://example.test", model="claude-sonnet-4-6")
    haiku = AnthropicModel(api_key="x", base_url="https://example.test", model="claude-haiku-4-5")
    unknown = AnthropicModel(api_key="x", base_url="https://example.test", model="some-future-model")

    assert sonnet.max_tokens == 64_000
    assert haiku.max_tokens == 16_384
    assert unknown.max_tokens == 32_768  # fallback


def test_anthropic_model_max_tokens_env_overrides_table(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "1234")
    model = AnthropicModel(api_key="x", base_url="https://example.test", model="claude-sonnet-4-6")
    assert model.max_tokens == 1234


def test_anthropic_model_max_tokens_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "1234")
    model = AnthropicModel(
        api_key="x",
        base_url="https://example.test",
        model="claude-sonnet-4-6",
        max_tokens=4321,
    )
    assert model.max_tokens == 4321
