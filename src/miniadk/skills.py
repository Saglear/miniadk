from __future__ import annotations

import ast
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .core.agent import Agent
from .core.tools import Tool, filter_tools, normalize_tool_name, tool


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"null", "none", "~"}:
        return None
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:  # noqa: BLE001 - keep parser forgiving
            parsed = [item.strip() for item in text[1:-1].split(",") if item.strip()]
        return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
    if "," in text:
        return [item.strip() for item in text.split(",") if item.strip()]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("_", "-")


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    end_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end_index = index
            break
    if end_index is None:
        return {}, text

    frontmatter_lines = lines[1:end_index]
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")

    data: dict[str, Any] = {}
    index = 0
    while index < len(frontmatter_lines):
        raw_line = frontmatter_lines[index].rstrip()
        stripped = raw_line.strip()
        index += 1

        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw_line:
            continue

        key, raw_value = raw_line.split(":", 1)
        key = _normalize_key(key)
        raw_value = raw_value.strip()

        if raw_value in {"|", ">"}:
            block: list[str] = []
            while index < len(frontmatter_lines):
                next_line = frontmatter_lines[index]
                if not next_line.strip():
                    block.append("")
                    index += 1
                    continue
                if next_line.startswith(" ") or next_line.startswith("\t"):
                    block.append(next_line.lstrip())
                    index += 1
                    continue
                break
            if raw_value == "|":
                data[key] = "\n".join(block).strip()
            else:
                data[key] = " ".join(part.strip() for part in block if part.strip())
            continue

        if raw_value == "":
            items: list[Any] = []
            lookahead = index
            while lookahead < len(frontmatter_lines):
                next_line = frontmatter_lines[lookahead]
                if next_line.strip().startswith("- "):
                    items.append(_parse_scalar(next_line.strip()[2:]))
                    lookahead += 1
                    continue
                if not next_line.strip():
                    lookahead += 1
                    continue
                break
            if items:
                data[key] = items
                index = lookahead
                continue
            data[key] = ""
            continue

        data[key] = _parse_scalar(raw_value)

    return data, body


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:  # noqa: BLE001 - forgiving parser
            parsed = [item.strip() for item in text[1:-1].split(",") if item.strip()]
        return _parse_list(parsed)
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_slash_command(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    command = stripped[1:]
    if not command:
        return None
    name, _, args = command.partition(" ")
    return name.strip(), args.strip()


@dataclass(slots=True)
class SkillInvocation:
    skill: str
    text: str
    allowed_tools: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True, slots=True)
class SkillProblem:
    skill: str
    message: str
    path: Path | None = None

    def __str__(self) -> str:
        if self.path is None:
            return f"{self.skill}: {self.message}"
        return f"{self.skill} ({self.path}): {self.message}"


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    body: str
    allowed_tools: list[str] = field(default_factory=list)
    when_to_use: str | None = None
    user_invocable: bool = True
    model_invocable: bool = True
    model: str | None = None
    effort: str | None = None
    context: str | None = None
    arguments: list[str] = field(default_factory=list)
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    problems: list[SkillProblem] = field(default_factory=list)

    @classmethod
    def from_markdown(cls, path: str | Path, text: str | None = None) -> "Skill":
        skill_path = Path(path)
        content = skill_path.read_text(encoding="utf-8") if text is None else text
        frontmatter, body = _parse_frontmatter(content)

        name = str(frontmatter.get("name") or skill_path.parent.name or skill_path.stem)
        description = str(frontmatter.get("description") or _first_line(body) or name)
        disable_model = bool(frontmatter.get("disable-model-invocation", False))

        skill = cls(
            name=name,
            description=description,
            body=body.strip(),
            allowed_tools=_parse_list(frontmatter.get("allowed-tools")),
            when_to_use=str(frontmatter.get("when_to_use") or frontmatter.get("when-to-use") or "") or None,
            user_invocable=bool(frontmatter.get("user-invocable", True)),
            model_invocable=not disable_model,
            model=str(frontmatter.get("model")) if frontmatter.get("model") not in {None, ""} else None,
            effort=str(frontmatter.get("effort")) if frontmatter.get("effort") not in {None, ""} else None,
            context=str(frontmatter.get("context")) if frontmatter.get("context") not in {None, ""} else None,
            arguments=_parse_list(frontmatter.get("arguments")),
            source_path=skill_path,
            metadata=frontmatter,
        )
        skill.problems.extend(_skill_problems(skill, frontmatter, content))
        return skill

    def render(
        self,
        args: str | dict[str, Any] = "",
        *,
        session_id: str | None = None,
    ) -> SkillInvocation:
        text = self.body
        raw_args = _args_text(args, self.arguments)
        named_args = args if isinstance(args, dict) else {}
        pieces = shlex.split(raw_args) if raw_args.strip() else []

        replacements = {
            "$ARGUMENTS": raw_args,
            "${ARGUMENTS}": raw_args,
            "$ARGS": raw_args,
            "${ARGS}": raw_args,
            "{{args}}": raw_args,
        }
        for token, value in replacements.items():
            text = text.replace(token, value)

        for index, piece in enumerate(pieces, start=1):
            text = text.replace(f"${index}", piece)

        for name, piece in zip(self.arguments, pieces):
            text = _replace_arg(text, name, piece)

        for name, value in named_args.items():
            text = _replace_arg(text, str(name), str(value))

        if self.source_path is not None:
            skill_dir = self.source_path.parent.as_posix()
            text = text.replace("${CLAUDE_SKILL_DIR}", skill_dir)
            text = f"Base directory for this skill: {skill_dir}\n\n{text}"

        if session_id:
            text = text.replace("${CLAUDE_SESSION_ID}", session_id)

        return SkillInvocation(
            skill=self.name,
            text=text.strip(),
            allowed_tools=_allowed_tool_names(self.allowed_tools),
        )


def _args_text(args: str | dict[str, Any], names: list[str]) -> str:
    if isinstance(args, dict):
        ordered = [str(args[name]) for name in names if name in args]
        if ordered:
            return " ".join(shlex.quote(item) for item in ordered)
        return " ".join(f"{key}={value}" for key, value in args.items())
    return args


def _replace_arg(text: str, name: str, value: str) -> str:
    text = text.replace(f"${name}", value)
    text = text.replace(f"${{{name}}}", value)
    return text.replace(f"{{{{{name}}}}}", value)


def skill(
    name: str,
    body: str,
    *,
    desc: str | None = None,
    tools: list[str] | tuple[str, ...] | str | None = None,
    args: list[str] | tuple[str, ...] | str | None = None,
    when: str | None = None,
    user: bool = True,
    model: bool = True,
) -> Skill:
    return Skill(
        name=name,
        description=desc or _first_line(body) or name,
        body=body,
        allowed_tools=_parse_list(tools),
        arguments=_parse_list(args),
        when_to_use=when,
        user_invocable=user,
        model_invocable=model,
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


@dataclass(slots=True)
class SkillRegistry:
    skills: list[Skill] = field(default_factory=list)

    @classmethod
    def from_skills(cls, *skills: Skill) -> "SkillRegistry":
        return cls(skills=cls._dedupe(list(skills)))

    @classmethod
    def from_paths(cls, *paths: str | Path) -> "SkillRegistry":
        skills: list[Skill] = []
        for root in paths:
            root_path = Path(root)
            if not root_path.exists():
                continue
            if root_path.is_file():
                skills.append(Skill.from_markdown(root_path))
                continue
            for skill_file in sorted(root_path.rglob("SKILL.md")):
                if skill_file.is_file():
                    skills.append(Skill.from_markdown(skill_file))
        return cls(skills=cls._dedupe(skills))

    @classmethod
    def from_claude_roots(
        cls,
        *,
        project_root: str | Path = ".",
        user_root: str | Path | None = None,
    ) -> "SkillRegistry":
        roots = [Path(project_root) / ".claude" / "skills"]
        if user_root is None:
            roots.append(Path.home() / ".claude" / "skills")
        else:
            roots.append(Path(user_root))
        return cls.from_paths(*roots)

    @staticmethod
    def _dedupe(skills: list[Skill]) -> list[Skill]:
        seen: dict[str, Skill] = {}
        for skill in skills:
            normalized = normalize_tool_name(skill.name)
            previous = seen.get(normalized)
            if previous is not None:
                skill.problems.append(
                    SkillProblem(
                        skill=skill.name,
                        message=f"duplicate skill name also used by {previous.name}",
                        path=skill.source_path,
                    )
                )
            seen[normalized] = skill
        return list(seen.values())

    def add(self, skill: Skill) -> None:
        normalized = normalize_tool_name(skill.name)
        self.skills = [existing for existing in self.skills if normalize_tool_name(existing.name) != normalized]
        self.skills.append(skill)

    def get(self, name: str) -> Skill | None:
        normalized = normalize_tool_name(name.lstrip("/"))
        for skill in reversed(self.skills):
            if normalize_tool_name(skill.name) == normalized:
                return skill
        return None

    def all(self) -> list[Skill]:
        return list(self.skills)

    def user_skills(self) -> list[Skill]:
        return [skill for skill in self.skills if skill.user_invocable]

    def model_skills(self) -> list[Skill]:
        return [skill for skill in self.skills if skill.model_invocable]

    def problems(self) -> list[SkillProblem]:
        problems: list[SkillProblem] = []
        seen: dict[str, Skill] = {}
        for skill in self.skills:
            problems.extend(skill.problems)
            normalized = normalize_tool_name(skill.name)
            if normalized in seen:
                problems.append(
                    SkillProblem(
                        skill=skill.name,
                        message=f"duplicate skill name also used by {seen[normalized].name}",
                        path=skill.source_path,
                    )
                )
            else:
                seen[normalized] = skill
        return problems

    def catalog_text(self) -> str:
        lines = []
        for skill in self.model_skills():
            details = skill.description
            if skill.when_to_use:
                details = f"{details} - {skill.when_to_use}"
            lines.append(f"- {skill.name}: {details}")
        return "\n".join(lines)

    def tool(self) -> Tool | None:
        model_skills = self.model_skills()
        if not model_skills:
            return None

        registry = self

        @tool(schema={"skill": _skill_schema(model_skills), "args": _skill_args_schema()})
        def skill(skill: str, args: str | dict = "") -> SkillInvocation:
            """Invoke a registered skill by name."""
            selected = registry.get(skill)
            if selected is None:
                raise ValueError(f"Unknown skill: {skill}")
            if not selected.model_invocable:
                raise ValueError(f"Skill {selected.name} cannot be used by the model")
            return selected.render(args)

        return skill


def tool_for_skill_registry(registry: SkillRegistry) -> Tool | None:
    return registry.tool()


def _skill_schema(skills: list[Skill]) -> dict[str, object]:
    schema: dict[str, object] = {
        "type": "string",
        "description": "Skill name to invoke.",
    }
    names = sorted(skill.name for skill in skills)
    if names:
        schema["enum"] = names
    return schema


def _skill_args_schema() -> dict[str, object]:
    return {
        "description": "Skill arguments as plain text or named argument object.",
        "oneOf": [
            {"type": "string"},
            {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        ],
    }


def _skill_problems(
    skill: Skill,
    frontmatter: dict[str, Any],
    content: str,
) -> list[SkillProblem]:
    problems: list[SkillProblem] = []
    if content.lstrip().startswith("---") and not frontmatter:
        problems.append(
            SkillProblem(
                skill=skill.name,
                message="frontmatter starts with --- but has no closing ---",
                path=skill.source_path,
            )
        )
    if not skill.body.strip():
        problems.append(
            SkillProblem(
                skill=skill.name,
                message="body is empty",
                path=skill.source_path,
            )
        )
    if not skill.description.strip():
        problems.append(
            SkillProblem(
                skill=skill.name,
                message="description is empty",
                path=skill.source_path,
            )
        )
    allowed = {
        "name",
        "description",
        "when-to-use",
        "when_to_use",
        "allowed-tools",
        "user-invocable",
        "disable-model-invocation",
        "model",
        "effort",
        "context",
        "arguments",
    }
    for key in sorted(frontmatter):
        if key not in allowed:
            problems.append(
                SkillProblem(
                    skill=skill.name,
                    message=f"unknown metadata key: {key}",
                    path=skill.source_path,
                )
            )
    return problems


def skill_tools(registry: SkillRegistry, tools: list[Tool]) -> list[Tool]:
    selected = registry.tool()
    if selected is None:
        return list(tools)
    return [selected, *tools]


def filter_tools_for_skill(
    tools: list[Tool],
    allowed_tools: list[str],
) -> list[Tool]:
    return filter_tools(tools, _allowed_tool_names(allowed_tools), keep_names={"skill"})


def _allowed_tool_names(allowed_tools: list[str]) -> list[str]:
    return [_allowed_tool_name(name) for name in allowed_tools]


def _allowed_tool_name(name: str) -> str:
    text = str(name).strip()
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.strip()


async def resolve_agent(agent: Agent) -> Agent:
    resolved_tools = list(agent.tools)
    instructions = agent.instructions.rstrip()
    resolved_skills = agent.skills

    if agent.mcp is not None:
        mcp_skills = await agent.mcp.skills()
        if mcp_skills.all():
            if resolved_skills is None:
                resolved_skills = mcp_skills
            else:
                resolved_skills = SkillRegistry.from_skills(
                    *resolved_skills.all(),
                    *mcp_skills.all(),
                )

    if resolved_skills is not None:
        skill_tool = resolved_skills.tool()
        if skill_tool is not None:
            resolved_tools = [skill_tool, *resolved_tools]
        catalog = resolved_skills.catalog_text()
        if catalog:
            instructions = f"{instructions}\n\nAvailable skills:\n{catalog}".strip()

    if agent.mcp is not None:
        resolved_tools.extend(await agent.mcp.tools())
        resource_tool = await agent.mcp.resource_tool()
        if resource_tool is not None:
            resolved_tools.append(resource_tool)

    return Agent(
        name=agent.name,
        instructions=instructions,
        tools=resolved_tools,
        skills=resolved_skills,
        mcp=agent.mcp,
    )


def tools_for_skill(agent: Agent, skill_name: str) -> list[Tool]:
    if agent.skills is None:
        return list(agent.tools)
    skill = agent.skills.get(skill_name)
    if skill is None:
        return list(agent.tools)
    return filter_tools_for_skill(agent.tools, skill.allowed_tools)
