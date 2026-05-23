import asyncio
import ast
import inspect
import re
import types
import dataclasses
from enum import Enum
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    NotRequired,
    Required,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

_TOOL_ALIASES = {
    "read": "read_file",
    "read_file": "read_file",
    "write": "write_file",
    "write_file": "write_file",
    "delete": "delete_file",
    "delete_file": "delete_file",
    "remove": "delete_file",
    "rm": "delete_file",
    "move": "move_file",
    "move_file": "move_file",
    "mv": "move_file",
    "rename": "move_file",
    "copy": "copy_file",
    "copy_file": "copy_file",
    "cp": "copy_file",
    "list": "list_files",
    "list_files": "list_files",
    "list_workspace_files": "list_files",
    "ls": "list_files",
    "glob": "glob_files",
    "glob_files": "glob_files",
    "glob_workspace_files": "glob_files",
    "grep": "search_text",
    "search": "search_text",
    "search_text": "search_text",
    "search_workspace_text": "search_text",
    "edit": "edit_file",
    "edit_file": "edit_file",
    "multi_edit": "edit_files",
    "multiedit": "edit_files",
    "edit_files": "edit_files",
    "bash": "shell",
    "shell": "shell",
    "task": "spawn_agent",
    "spawn": "spawn_agent",
    "spawn_agent": "spawn_agent",
    "todo": "todo_write",
    "todo_write": "todo_write",
    "todowrite": "todo_write",
    "todo_read": "todo_read",
    "todoread": "todo_read",
    "start_work": "start_work",
    "startwork": "start_work",
    "list_work": "list_work",
    "listwork": "list_work",
    "worklist": "list_work",
    "read_work": "read_work",
    "readwork": "read_work",
    "cancel_work": "cancel_work",
    "cancelwork": "cancel_work",
    "fetch": "fetch_url",
    "fetch_url": "fetch_url",
    "url_fetch": "fetch_url",
    "webfetch": "fetch_url",
    "web_fetch": "fetch_url",
    "skill": "skill",
}


def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Signature.empty:
        return {"type": "any"}
    if isinstance(annotation, str):
        return _schema_for_annotation_text(annotation)
    if annotation is Any:
        return {"type": "any"}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation in (list, tuple, set):
        return {"type": "array"}
    if annotation is dict:
        return {"type": "object"}
    if dataclasses.is_dataclass(annotation):
        return _schema_for_dataclass(annotation)
    if _is_typeddict(annotation):
        return _schema_for_typeddict(annotation)

    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        schema = _schema_for_annotation(args[0])
        schema.update(_annotation_schema(args[1:]))
        description = _annotation_description(args[1:])
        if description is not None:
            schema["description"] = description
        return schema
    if origin is Literal:
        return _schema_for_enum_values(get_args(annotation))
    if origin in {Union, types.UnionType}:
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return _schema_for_annotation(args[0])
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        return _schema_for_enum_values(member.value for member in annotation)
    if origin in (list, tuple, set):
        args = get_args(annotation)
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = _schema_for_annotation(args[0])
        return schema
    if origin is dict:
        args = get_args(annotation)
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _schema_for_annotation(args[1])
        return schema

    return {"type": getattr(annotation, "__name__", "string")}


def _schema_for_dataclass(cls: type) -> dict[str, Any]:
    type_hints = _type_hints_for_dataclass(cls)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for field in dataclasses.fields(cls):
        annotation = type_hints.get(field.name, field.type)
        field_schema = _schema_for_annotation(annotation)
        description = _field_description(field)
        if description is not None and "description" not in field_schema:
            field_schema["description"] = description
        if _is_json_value(field.default):
            field_schema["default"] = field.default
        properties[field.name] = field_schema
        if (
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING
        ):
            required.append(field.name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _is_typeddict(annotation: Any) -> bool:
    return (
        inspect.isclass(annotation)
        and isinstance(getattr(annotation, "__required_keys__", None), frozenset)
        and isinstance(getattr(annotation, "__optional_keys__", None), frozenset)
    )


def _schema_for_typeddict(cls: type) -> dict[str, Any]:
    type_hints = _type_hints_for_dataclass(cls)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    required_keys = set(getattr(cls, "__required_keys__", frozenset()))

    for name, annotation in type_hints.items():
        unwrapped = _unwrap_required_annotation(annotation)
        properties[name] = _schema_for_annotation(unwrapped)
        requirement = _required_annotation_kind(annotation)
        if requirement == "required" or (
            requirement is None and name in required_keys
        ):
            required.append(name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def _unwrap_required_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in {Required, NotRequired}:
        args = get_args(annotation)
        if args:
            return args[0]
    return annotation


def _required_annotation_kind(annotation: Any) -> str | None:
    origin = get_origin(annotation)
    if origin is Required:
        return "required"
    if origin is NotRequired:
        return "not_required"
    return None


def _type_hints_for_dataclass(cls: type) -> dict[str, Any]:
    try:
        return get_type_hints(cls, include_extras=True)
    except (AttributeError, NameError, SyntaxError, TypeError, ValueError):
        return {}


def _schema_for_enum_values(values: Any) -> dict[str, Any]:
    enum = list(values)
    schema: dict[str, Any] = {"enum": enum}
    enum_type = _enum_value_type(enum)
    if enum_type is not None:
        schema["type"] = enum_type
    return schema


def _enum_value_type(values: list[Any]) -> str | None:
    if not values:
        return None
    if all(isinstance(value, str) for value in values):
        return "string"
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return "number"
    return None


def _schema_for_annotation_text(annotation: str) -> dict[str, Any]:
    stripped = annotation.strip()
    if stripped.lower().startswith("literal[") and stripped.endswith("]"):
        inner = stripped[stripped.index("[") + 1 : -1]
        try:
            return _schema_for_enum_values(ast.literal_eval(f"({inner},)"))
        except (SyntaxError, ValueError):
            return {"type": "string"}
    normalized = annotation.strip().lower()
    if "|" in normalized:
        parts = [
            part.strip()
            for part in normalized.split("|")
            if part.strip() not in {"none", "null", "nonetype"}
        ]
        if len(parts) == 1:
            return _schema_for_annotation_text(parts[0])
    if normalized.startswith("optional[") and normalized.endswith("]"):
        return _schema_for_annotation_text(
            normalized.removeprefix("optional[").removesuffix("]")
        )
    if normalized in {"str", "string"}:
        return {"type": "string"}
    if normalized in {"int", "integer"}:
        return {"type": "integer"}
    if normalized in {"float", "number"}:
        return {"type": "number"}
    if normalized in {"bool", "boolean"}:
        return {"type": "boolean"}
    if normalized in {"list", "tuple", "set", "array"}:
        return {"type": "array"}
    if normalized.startswith(("list[", "tuple[", "set[")) and normalized.endswith("]"):
        inner = normalized[normalized.index("[") + 1 : -1]
        return {"type": "array", "items": _schema_for_annotation_text(inner)}
    if normalized in {"dict", "mapping", "object"}:
        return {"type": "object"}
    if normalized.startswith(("dict[", "mapping[")) and normalized.endswith("]"):
        inner = normalized[normalized.index("[") + 1 : -1]
        pieces = [piece.strip() for piece in inner.split(",", 1)]
        schema: dict[str, Any] = {"type": "object"}
        if len(pieces) == 2:
            schema["additionalProperties"] = _schema_for_annotation_text(pieces[1])
        return schema
    return {"type": "string"}


def _type_name(annotation: Any) -> str:
    return str(_schema_for_annotation(annotation).get("type", "string"))


def _schema_from_function(
    fn: Callable,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signature = inspect.signature(fn)
    type_hints = _type_hints_for_function(fn)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if name == "progress":
            continue
        annotation = type_hints.get(name, parameter.annotation)
        field_schema = _schema_for_annotation(annotation)
        if _is_json_value(parameter.default):
            field_schema["default"] = parameter.default
        properties[name] = field_schema
        if parameter.default is inspect.Signature.empty:
            required.append(name)

    if schema:
        for name, value in schema.items():
            if name in properties and isinstance(value, dict):
                properties[name] = value

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        result["required"] = required
    return result


def _type_hints_for_function(fn: Callable) -> dict[str, Any]:
    try:
        return get_type_hints(fn, include_extras=True)
    except (AttributeError, NameError, SyntaxError, TypeError, ValueError):
        return {}


def _annotation_description(metadata: tuple[Any, ...]) -> str | None:
    for item in metadata:
        if isinstance(item, str) and item.strip():
            return item
    return None


def _annotation_schema(metadata: tuple[Any, ...]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for item in metadata:
        if isinstance(item, dict):
            schema.update(item)
    return schema


def _field_description(field: dataclasses.Field) -> str | None:
    for key in ("description", "doc"):
        value = field.metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _is_json_value(value: Any) -> bool:
    if value is inspect.Signature.empty or value is dataclasses.MISSING:
        return False
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


@dataclass(slots=True)
class ToolValidation:
    ok: bool
    message: str | None = None

    @classmethod
    def allow(cls) -> "ToolValidation":
        return cls(True)

    @classmethod
    def deny(cls, message: str) -> "ToolValidation":
        return cls(False, message)


@dataclass(slots=True)
class ToolMeta:
    read_only: bool | Callable[..., bool] = False
    destructive: bool | Callable[..., bool] = False
    concurrency_safe: bool | Callable[..., bool] = False
    validate: Callable[..., Any] | None = None
    format: Callable[..., Any] | None = None
    max_text: int | None = None
    timeout: float | None = None
    schema: dict[str, Any] | None = None

    async def validate_input(self, **kwargs: Any) -> ToolValidation:
        if self.validate is None:
            return ToolValidation.allow()

        result = _call_metadata(self.validate, **kwargs)
        if inspect.isawaitable(result):
            result = await result

        if isinstance(result, ToolValidation):
            return result
        if result is None or result is True:
            return ToolValidation.allow()
        if result is False:
            return ToolValidation.deny("Tool input failed validation")
        if isinstance(result, str):
            return ToolValidation.deny(result)
        return ToolValidation.allow() if bool(result) else ToolValidation.deny(
            "Tool input failed validation"
        )

    def is_read_only(self, **kwargs: Any) -> bool:
        return self._flag(self.read_only, **kwargs)

    def is_destructive(self, **kwargs: Any) -> bool:
        return self._flag(self.destructive, **kwargs)

    def is_concurrency_safe(self, **kwargs: Any) -> bool:
        return self._flag(self.concurrency_safe, **kwargs)

    async def text(self, result: Any, **kwargs: Any) -> str:
        if self.format is None:
            text = result
        else:
            text = _call_format(self.format, result, kwargs)
            if inspect.isawaitable(text):
                text = await text
        return _clip_text(str(text), self.max_text)

    @staticmethod
    def _flag(value: bool | Callable[..., bool], **kwargs: Any) -> bool:
        if callable(value):
            return bool(_call_metadata(value, **kwargs))
        return bool(value)


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable
    permission: Any = None
    meta: ToolMeta | None = None
    annotations: dict[str, Any] | None = None

    async def run(self, **kwargs: Any) -> Any:
        result = self.func(**self._call_kwargs(self._coerce_kwargs(kwargs)))
        if inspect.isawaitable(result):
            timeout = self.meta.timeout if self.meta is not None else None
            if timeout is not None:
                try:
                    return await asyncio.wait_for(result, timeout=timeout)
                except TimeoutError as error:
                    raise TimeoutError(
                        f"{self.name} timed out after {timeout:g} seconds"
                    ) from error
            return await result
        return result

    def __call__(self, **kwargs: Any) -> Any:
        return self.func(**self._call_kwargs(self._coerce_kwargs(kwargs)))

    async def validate(self, **kwargs: Any) -> ToolValidation:
        schema_validation = _validate_tool_schema(self.input_schema, kwargs)
        if not schema_validation.ok:
            return schema_validation
        if self.meta is None:
            return ToolValidation.allow()
        return await self.meta.validate_input(**kwargs)

    def is_read_only(self, **kwargs: Any) -> bool:
        return self.meta.is_read_only(**kwargs) if self.meta is not None else False

    def is_destructive(self, **kwargs: Any) -> bool:
        return self.meta.is_destructive(**kwargs) if self.meta is not None else False

    def is_concurrency_safe(self, **kwargs: Any) -> bool:
        if self.meta is None:
            return False
        return self.meta.is_concurrency_safe(**kwargs)

    async def text(self, result: Any, **kwargs: Any) -> str:
        if self.meta is None:
            return str(result)
        return await self.meta.text(result, **kwargs)

    def _call_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "progress" not in kwargs:
            return kwargs
        if "progress" in inspect.signature(self.func).parameters:
            return kwargs
        return {
            name: value
            for name, value in kwargs.items()
            if name != "progress"
        }

    def _coerce_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if not self.annotations:
            return kwargs
        return {
            name: _coerce_value(self.annotations.get(name), value)
            for name, value in kwargs.items()
        }


def _build_tool(
    fn: Callable,
    permission: Any = None,
    meta: ToolMeta | None = None,
) -> Tool:
    description = inspect.getdoc(fn) or ""
    annotations = _type_hints_for_function(fn)
    built = Tool(
        name=fn.__name__,
        description=description,
        input_schema=_schema_from_function(
            fn,
            schema=meta.schema if meta is not None else None,
        ),
        func=fn,
        permission=permission,
        meta=meta,
        annotations=annotations,
    )
    return built


def _call_format(fn: Callable, result: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(result, **kwargs)

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return fn(result, **kwargs)

    known = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return fn(result, **known)


def _call_metadata(fn: Callable, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return fn(**kwargs)

    known = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return fn(**known)


def _coerce_value(annotation: Any, value: Any) -> Any:
    if annotation is None or annotation is inspect.Signature.empty:
        return value
    if isinstance(annotation, str):
        return value
    if (
        inspect.isclass(annotation)
        and issubclass(annotation, Enum)
        and not isinstance(value, annotation)
    ):
        return annotation(value)
    if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
        type_hints = _type_hints_for_dataclass(annotation)
        return annotation(
            **{
                key: _coerce_value(type_hints.get(key), item)
                for key, item in value.items()
            }
        )

    origin = get_origin(annotation)
    if origin in {Union, types.UnionType}:
        args = [item for item in get_args(annotation) if item is not type(None)]
        if len(args) == 1:
            return _coerce_value(args[0], value)
    if origin is list and isinstance(value, list):
        args = get_args(annotation)
        if not args:
            return value
        inner = args[0]
        return [_coerce_value(inner, item) for item in value]
    if origin is tuple and isinstance(value, list):
        args = get_args(annotation)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_value(args[0], item) for item in value)
        if args:
            return tuple(
                _coerce_value(args[min(index, len(args) - 1)], item)
                for index, item in enumerate(value)
            )
        return tuple(value)
    if origin is set and isinstance(value, list):
        args = get_args(annotation)
        if not args:
            return set(value)
        inner = args[0]
        return {_coerce_value(inner, item) for item in value}
    if origin is dict and isinstance(value, dict):
        args = get_args(annotation)
        if len(args) != 2:
            return value
        return {
            key: _coerce_value(args[1], item)
            for key, item in value.items()
        }
    return value


def _validate_tool_schema(schema: dict[str, Any], arguments: dict[str, Any]) -> ToolValidation:
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return ToolValidation.allow()

    for name in schema.get("required", []):
        if name not in arguments:
            return ToolValidation.deny(f"Missing required tool argument: {name}")

    if schema.get("additionalProperties") is False:
        for name in arguments:
            if name not in properties:
                return ToolValidation.deny(f"Unknown tool argument: {name}")

    for name, value in arguments.items():
        field = properties.get(name)
        if isinstance(field, dict):
            validation = _validate_schema_value(field, value, name)
            if not validation.ok:
                return validation

    return ToolValidation.allow()


def _validate_schema_value(
    schema: dict[str, Any],
    value: Any,
    path: str,
) -> ToolValidation:
    for key in ("oneOf", "anyOf"):
        options = schema.get(key)
        if isinstance(options, list):
            if any(
                isinstance(option, dict)
                and _validate_schema_value(option, value, path).ok
                for option in options
            ):
                return ToolValidation.allow()
            return ToolValidation.deny(f"Tool argument {path} must match {key} schema")

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        allowed = ", ".join(str(item) for item in enum)
        return ToolValidation.deny(f"Tool argument {path} must be one of: {allowed}")

    if "const" in schema and value != schema["const"]:
        return ToolValidation.deny(f"Tool argument {path} must be {schema['const']}")

    schema_type = schema.get("type")
    if schema_type in {None, "any"}:
        return ToolValidation.allow()
    if schema_type == "string" and not isinstance(value, str):
        return ToolValidation.deny(f"Tool argument {path} must be string")
    if schema_type == "string":
        validation = _validate_string_schema(schema, value, path)
        if not validation.ok:
            return validation
    if schema_type == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        return ToolValidation.deny(f"Tool argument {path} must be integer")
    if schema_type == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        return ToolValidation.deny(f"Tool argument {path} must be number")
    if schema_type in {"integer", "number"}:
        validation = _validate_number_schema(schema, value, path)
        if not validation.ok:
            return validation
    if schema_type == "boolean" and not isinstance(value, bool):
        return ToolValidation.deny(f"Tool argument {path} must be boolean")
    if schema_type == "array":
        if not isinstance(value, list):
            return ToolValidation.deny(f"Tool argument {path} must be array")
        validation = _validate_array_schema(schema, value, path)
        if not validation.ok:
            return validation
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                validation = _validate_schema_value(item_schema, item, f"{path}[{index}]")
                if not validation.ok:
                    return validation
        return ToolValidation.allow()
    if schema_type == "object":
        if not isinstance(value, dict):
            return ToolValidation.deny(f"Tool argument {path} must be object")
        validation = _validate_object_schema(schema, value, path)
        if not validation.ok:
            return validation
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name in schema.get("required", []):
                if name not in value:
                    return ToolValidation.deny(
                        f"Missing required tool argument: {path}.{name}"
                    )
            if schema.get("additionalProperties") is False:
                for name in value:
                    if name not in properties:
                        return ToolValidation.deny(
                            f"Unknown tool argument: {path}.{name}"
                        )
            for name, item in value.items():
                field = properties.get(name)
                if isinstance(field, dict):
                    validation = _validate_schema_value(field, item, f"{path}.{name}")
                    if not validation.ok:
                        return validation
            return ToolValidation.allow()
        value_schema = schema.get("additionalProperties")
        if isinstance(value_schema, dict):
            for key, item in value.items():
                validation = _validate_schema_value(value_schema, item, f"{path}.{key}")
                if not validation.ok:
                    return validation
        return ToolValidation.allow()
    return ToolValidation.allow()


def _validate_string_schema(
    schema: dict[str, Any],
    value: str,
    path: str,
) -> ToolValidation:
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        return ToolValidation.deny(f"Tool argument {path} must be at least {min_length} chars")
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        return ToolValidation.deny(f"Tool argument {path} must be at most {max_length} chars")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and re.search(pattern, value) is None:
        return ToolValidation.deny(f"Tool argument {path} must match pattern: {pattern}")
    return ToolValidation.allow()


def _validate_number_schema(
    schema: dict[str, Any],
    value: int | float,
    path: str,
) -> ToolValidation:
    minimum = schema.get("minimum")
    if isinstance(minimum, (int, float)) and value < minimum:
        return ToolValidation.deny(f"Tool argument {path} must be >= {minimum}")
    maximum = schema.get("maximum")
    if isinstance(maximum, (int, float)) and value > maximum:
        return ToolValidation.deny(f"Tool argument {path} must be <= {maximum}")
    exclusive_minimum = schema.get("exclusiveMinimum")
    if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
        return ToolValidation.deny(
            f"Tool argument {path} must be > {exclusive_minimum}"
        )
    exclusive_maximum = schema.get("exclusiveMaximum")
    if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
        return ToolValidation.deny(
            f"Tool argument {path} must be < {exclusive_maximum}"
        )
    multiple_of = schema.get("multipleOf")
    if (
        isinstance(multiple_of, (int, float))
        and not isinstance(multiple_of, bool)
        and multiple_of > 0
    ):
        quotient = value / multiple_of
        if abs(quotient - round(quotient)) > 1e-9:
            return ToolValidation.deny(
                f"Tool argument {path} must be a multiple of {multiple_of}"
            )
    return ToolValidation.allow()


def _validate_array_schema(
    schema: dict[str, Any],
    value: list[Any],
    path: str,
) -> ToolValidation:
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        return ToolValidation.deny(f"Tool argument {path} must have at least {min_items} items")
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        return ToolValidation.deny(f"Tool argument {path} must have at most {max_items} items")
    if schema.get("uniqueItems") is True:
        for index, item in enumerate(value):
            if item in value[:index]:
                return ToolValidation.deny(
                    f"Tool argument {path} must have unique items"
                )
    return ToolValidation.allow()


def _validate_object_schema(
    schema: dict[str, Any],
    value: dict[str, Any],
    path: str,
) -> ToolValidation:
    min_properties = schema.get("minProperties")
    if isinstance(min_properties, int) and len(value) < min_properties:
        return ToolValidation.deny(
            f"Tool argument {path} must have at least {min_properties} properties"
        )
    max_properties = schema.get("maxProperties")
    if isinstance(max_properties, int) and len(value) > max_properties:
        return ToolValidation.deny(
            f"Tool argument {path} must have at most {max_properties} properties"
        )
    return ToolValidation.allow()


def _clip_text(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    suffix = "\n...[truncated]"
    if limit <= len(suffix):
        return text[:limit]
    return text[: limit - len(suffix)] + suffix


def normalize_tool_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_tool_name(name: str) -> str:
    return _TOOL_ALIASES.get(normalize_tool_name(name), normalize_tool_name(name))


def tool_matches_name(tool: Tool, name: str) -> bool:
    return canonical_tool_name(tool.name) == canonical_tool_name(name)


def filter_tools(
    tools: list[Tool],
    allowed_names: list[str] | None,
    *,
    keep_names: set[str] | None = None,
) -> list[Tool]:
    if not allowed_names:
        return list(tools)

    keep = {canonical_tool_name(name) for name in (keep_names or set())}
    allowed = {canonical_tool_name(name) for name in allowed_names}
    filtered: list[Tool] = []
    seen: set[str] = set()
    for tool in tools:
        canonical = canonical_tool_name(tool.name)
        if canonical in keep or canonical in allowed:
            if tool.name not in seen:
                filtered.append(tool)
                seen.add(tool.name)
    return filtered


def tool(
    fn: Callable | None = None,
    *,
    permission: Any = None,
    read_only: bool | Callable[..., bool] = False,
    destructive: bool | Callable[..., bool] = False,
    concurrency_safe: bool | Callable[..., bool] = False,
    validate: Callable[..., Any] | None = None,
    format: Callable[..., Any] | None = None,
    max_text: int | None = None,
    timeout: float | None = None,
    schema: dict[str, Any] | None = None,
):
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout must be > 0")
    meta = ToolMeta(
        read_only=read_only,
        destructive=destructive,
        concurrency_safe=concurrency_safe,
        validate=validate,
        format=format,
        max_text=max_text,
        timeout=timeout,
        schema=schema,
    )
    if fn is None:
        return lambda real_fn: _build_tool(
            real_fn,
            permission=permission,
            meta=meta,
        )
    return _build_tool(fn, permission=permission, meta=meta)
