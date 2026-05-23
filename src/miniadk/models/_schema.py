from typing import Any

from ..core.tools import Tool


_FIELD_KEYS = {
    "type",
    "description",
    "enum",
    "items",
    "properties",
    "additionalProperties",
    "oneOf",
    "anyOf",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
}


def tool_parameters(tool: Tool) -> dict[str, Any]:
    schema = dict(tool.input_schema)
    if schema.get("type") == "object" and "properties" in schema:
        return _provider_schema(schema)

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, field in schema.items():
        if not isinstance(field, dict):
            continue
        properties[name] = _field_schema(field)
        if field.get("required", False):
            required.append(name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _field_schema(field: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: value
        for key, value in field.items()
        if key in _FIELD_KEYS
    }
    if "type" not in result and "oneOf" not in result and "anyOf" not in result:
        result["type"] = "any"
    return _provider_schema(result)


def _provider_schema(schema: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and value == "any":
            continue
        if key in {"properties", "additionalProperties"}:
            result[key] = _provider_nested_schema(value)
            continue
        if key in {"items"}:
            result[key] = _provider_nested_schema(value)
            continue
        if key in {"oneOf", "anyOf"} and isinstance(value, list):
            result[key] = [
                _provider_schema(item)
                for item in value
                if isinstance(item, dict)
            ]
            continue
        result[key] = value
    return result


def _provider_nested_schema(value: Any) -> Any:
    if isinstance(value, dict):
        if _looks_like_schema(value):
            return _provider_schema(value)
        return {
            key: _provider_nested_schema(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_provider_nested_schema(item) for item in value]
    return value


def _looks_like_schema(value: dict[str, Any]) -> bool:
    return any(key in value for key in _FIELD_KEYS)
