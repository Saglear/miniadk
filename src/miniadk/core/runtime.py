import asyncio
from collections.abc import AsyncGenerator
from uuid import uuid4

from .agent import Agent
from .events import Event
from .messages import Message
from .middleware import (
    Middleware,
    PermissionRequest,
    ask_allowed,
    current_ask_user,
    middleware_ask_user,
    pop_ask_user,
    push_ask_user,
)
from .model import Model, ToolCall
from .policy import DefaultRunPolicy, RunDecision, RunPolicy, RunState
from .session import Session
from .tools import Tool, filter_tools, tool_matches_name


class Runtime:
    def __init__(
        self,
        agent: Agent,
        model: Model,
        middleware: list[Middleware] | None = None,
        policy: RunPolicy | None = None,
        session: Session | None = None,
        max_steps: int = 20,
    ):
        self.agent = agent
        self.model = model
        self.middleware = middleware or []
        self.policy = policy or DefaultRunPolicy()
        self.session = session if session is not None else Session()
        if not self.session.messages:
            self.session.messages.append(Message("system", agent.instructions))
        self.max_steps = max_steps
        self._cancel_requested = False
        self._cancel_reason = "cancelled"
        self._cancel_signal: asyncio.Event | None = None

    @property
    def messages(self) -> list[Message]:
        return self.session.messages

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancel_requested = True
        self._cancel_reason = reason or "cancelled"
        if self._cancel_signal is not None:
            self._cancel_signal.set()

    async def run(
        self,
        user_input: str,
        *,
        tools: list[Tool] | None = None,
        lifecycle: bool = False,
        trace: bool = False,
    ) -> AsyncGenerator[Event, None]:
        self._run_id = uuid4().hex
        self._trace_events = trace
        self._current_step = None
        self._stop_current_run = False
        self._cancel_requested = False
        self._cancel_reason = "cancelled"
        self._cancel_signal = asyncio.Event()
        ask_user_token = push_ask_user(middleware_ask_user(self.middleware))
        if lifecycle:
            yield self._event(
                "run_start",
                {
                    "agent": self.agent.name,
                    "input": user_input,
                },
            )
        self._run_status = "completed"
        self._run_reason = "completed"
        try:
            async for event in self._run_events(user_input, tools=tools):
                if event.type in {"error", "tool_error"}:
                    self._run_status = "error"
                    self._run_reason = str(event.data.get("reason") or event.type)
                elif event.type in {"cancelled", "tool_denied", "tool_invalid"}:
                    self._run_status = "stopped"
                    self._run_reason = str(event.data.get("reason") or event.type)
                yield event
        except Exception:
            if self._run_status == "completed":
                self._run_status = "error"
                self._run_reason = "error"
            if lifecycle:
                yield self._run_end_event()
            raise
        finally:
            pop_ask_user(ask_user_token)
        if lifecycle:
            yield self._run_end_event()

    async def _run_events(
        self,
        user_input: str,
        *,
        tools: list[Tool] | None = None,
    ) -> AsyncGenerator[Event, None]:
        self.messages.append(Message("user", user_input))
        active_tools = list(self.agent.tools if tools is None else tools)

        for step in range(1, self.max_steps + 1):
            self._current_step = step
            cancelled = self._cancel_event()
            if cancelled is not None:
                yield cancelled
                return
            result = None
            streamed_text = False
            stream_text_parts: list[str] = []
            stream = getattr(self.model, "stream", None)
            model_state = self._state(step, active_tools, None)
            await self._notify_before_model(model_state)
            try:
                if stream is not None:
                    async for update in stream(messages=self.messages, tools=active_tools):
                        cancelled = self._cancel_event()
                        if cancelled is not None:
                            self._save_stream_text(stream_text_parts)
                            yield cancelled
                            return
                        if update.delta:
                            streamed_text = True
                            stream_text_parts.append(update.delta)
                            yield self._event(
                                "message_delta",
                                {"text": update.delta},
                                step=step,
                            )
                        if update.thinking:
                            yield self._event(
                                "thinking_delta",
                                {"text": update.thinking},
                                step=step,
                            )
                        if update.tool_call is not None:
                            payload = {"index": update.tool_call.index}
                            if update.tool_call.id is not None:
                                payload["id"] = update.tool_call.id
                            if update.tool_call.name is not None:
                                payload["name"] = update.tool_call.name
                            if update.tool_call.arguments is not None:
                                payload["arguments"] = update.tool_call.arguments
                            yield self._event("tool_call_delta", payload, step=step)
                        if update.result is not None:
                            result = update.result
                        cancelled = self._cancel_event()
                        if cancelled is not None:
                            self._save_stream_text(stream_text_parts)
                            yield cancelled
                            return
                else:
                    result = await self.model.complete(
                        messages=self.messages,
                        tools=active_tools,
                    )
                    cancelled = self._cancel_event()
                    if cancelled is not None:
                        yield cancelled
                        return

                if result is None:
                    complete = getattr(self.model, "complete", None)
                    if complete is None:
                        yield self._event(
                            "error",
                            {
                                "message": "Streaming model ended without a final result",
                                "reason": "stream_missing_result",
                            },
                        )
                        self._save_stream_text(stream_text_parts)
                        return
                    result = await self.model.complete(
                        messages=self.messages,
                        tools=active_tools,
                    )
                    cancelled = self._cancel_event()
                    if cancelled is not None:
                        yield cancelled
                        return
            except Exception as error:
                self._save_stream_text(stream_text_parts)
                await self._notify_model_error(model_state, error)
                self._run_status = "error"
                self._run_reason = "model_error"
                raise

            await self._notify_after_model(self._state(step, active_tools, result))

            if result.tool_calls:
                assistant_text = (
                    result.message
                    if result.message is not None
                    else "".join(stream_text_parts)
                )
                self.messages.append(
                    Message(
                        "assistant",
                        assistant_text,
                        tool_calls=list(result.tool_calls),
                        content_blocks=result.content_blocks,
                    )
                )
            elif result.message is not None:
                self.messages.append(
                    Message(
                        "assistant",
                        result.message,
                        content_blocks=result.content_blocks,
                    )
                )
            elif stream_text_parts:
                self._save_stream_text(stream_text_parts)

            state = self._state(step, active_tools, result)
            decision = await self.policy.after_model(state)
            self._apply_decision(decision)
            if decision.action == "stop":
                self._run_status = _decision_status(decision)
                self._run_reason = str(decision.reason)
                if decision.message is not None:
                    data = {"text": decision.message}
                    if streamed_text:
                        data["streamed"] = True
                    yield self._event("message", data, step=step)
                return

            async for event in self._run_tool_calls(result.tool_calls, active_tools):
                if event.type == "tool_result":
                    allowed_tools = getattr(event.data.get("result"), "allowed_tools", None)
                    if allowed_tools:
                        active_tools = filter_tools(active_tools, allowed_tools)
                yield event
            if self._stop_current_run:
                return

            if result.tool_calls:
                state = self._state(step, active_tools, result)
                decision = await self.policy.after_tools(state)
                self._apply_decision(decision)
                if decision.action == "stop":
                    self._run_status = _decision_status(decision)
                    self._run_reason = str(decision.reason)
                    if decision.message is not None:
                        yield self._event(
                            "message",
                            {"text": decision.message},
                            step=step,
                        )
                    return

        self._run_status = "error"
        self._run_reason = "max_steps"
        yield self._event(
            "error",
            {
                "message": f"Runtime stopped after {self.max_steps} steps",
                "reason": "max_steps",
            },
        )

    async def ask(
        self,
        user_input: str,
        *,
        tools: list[Tool] | None = None,
    ) -> str:
        answer = ""
        async for event in self.run(user_input, tools=tools):
            if event.type == "message":
                answer = event.data["text"]
        return answer

    def _save_stream_text(self, parts: list[str]) -> None:
        if parts:
            self.messages.append(Message("assistant", "".join(parts)))
            parts.clear()

    async def _run_tool_calls(
        self,
        calls: list[ToolCall],
        tools: list[Tool],
    ) -> AsyncGenerator[Event, None]:
        if not self._can_run_concurrently(calls, tools):
            for call in calls:
                async for event in self._run_tool_call(call, tools):
                    yield event
                if self._stop_current_run:
                    return
            return

        prepared: list[tuple[ToolCall, Tool]] = []
        for call in calls:
            tool = self._find_tool(call.name, tools)
            if tool is None:
                async for event in self._unknown_tool(call):
                    yield event
                return

            invalid = await self._invalid_tool_input(tool, call)
            if invalid is not None:
                yield invalid
                return

            allowed, permission_events = await self._check_permissions(tool, call)
            for event in permission_events:
                yield event
            if not allowed:
                self._stop_current_run = True
                return

            prepared.append((call, tool))

        for call, tool in prepared:
            yield self._tool_event("tool_call", call, tool)

        tasks = [
            asyncio.create_task(self._run_concurrent_tool(call, tool))
            for call, tool in prepared
        ]
        gather_task = asyncio.gather(*tasks)
        cancel_task = asyncio.create_task(self._cancel_signal.wait())
        try:
            done, _ = await asyncio.wait(
                {gather_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done:
                gather_task.cancel()
                try:
                    await gather_task
                except asyncio.CancelledError:
                    pass
                cancelled = self._cancel_event()
                if cancelled is not None:
                    yield cancelled
                return

            cancel_task.cancel()
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass
            outcomes = await gather_task
        except BaseException:
            gather_task.cancel()
            cancel_task.cancel()
            await asyncio.gather(
                gather_task,
                cancel_task,
                return_exceptions=True,
            )
            raise

        for (call, tool), outcome in zip(prepared, outcomes):
            for event in outcome["events"]:
                yield self._event(event.type, event.data)

            error = outcome.get("error")
            if error is not None:
                self._stop_current_run = True
                message = f"{type(error).__name__}: {error}"
                self.messages.append(
                    Message("tool", message, name=tool.name, tool_call_id=call.id)
                )
                yield self._tool_event(
                    "tool_error",
                    call,
                    tool,
                    {
                        "tool": tool.name,
                        "message": message,
                    },
                )
                return

            text_result = outcome["text"]
            self.messages.append(
                Message("tool", text_result, name=tool.name, tool_call_id=call.id)
            )
            await self._notify_after_tool(
                tool,
                call.arguments,
                outcome["result"],
                text_result,
            )
            yield self._tool_event(
                "tool_result",
                call,
                tool,
                {
                    "name": tool.name,
                    "result": outcome["result"],
                    "text": text_result,
                },
            )

    async def _run_tool_call(
        self,
        call: ToolCall,
        tools: list[Tool],
    ) -> AsyncGenerator[Event, None]:
        cancelled = self._cancel_event()
        if cancelled is not None:
            yield cancelled
            return

        tool = self._find_tool(call.name, tools)
        if tool is None:
            async for event in self._unknown_tool(call):
                yield event
            return

        invalid = await self._invalid_tool_input(tool, call)
        if invalid is not None:
            yield invalid
            return

        allowed, permission_events = await self._check_permissions(tool, call)
        for event in permission_events:
            yield event
        if not allowed:
            self._stop_current_run = True
            return

        yield self._tool_event("tool_call", call, tool)
        progress = _Progress(tool.name)
        tool_task = asyncio.create_task(tool.run(**call.arguments, progress=progress))
        progress_task = asyncio.create_task(progress.next())
        cancel_task = asyncio.create_task(self._cancel_signal.wait())

        try:
            while True:
                done, _ = await asyncio.wait(
                    {tool_task, progress_task, cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task in done:
                    tool_task.cancel()
                    progress.close()
                    progress_task.cancel()
                    await asyncio.gather(
                        tool_task,
                        progress_task,
                        return_exceptions=True,
                    )
                    cancelled = self._cancel_event()
                    if cancelled is not None:
                        yield cancelled
                    return
                if progress_task in done:
                    event = progress_task.result()
                    if event is not None:
                        yield self._event(event.type, event.data)
                    progress_task = asyncio.create_task(progress.next())
                    cancelled = self._cancel_event()
                    if cancelled is not None:
                        tool_task.cancel()
                        progress.close()
                        progress_task.cancel()
                        cancel_task.cancel()
                        await asyncio.gather(
                            tool_task,
                            progress_task,
                            cancel_task,
                            return_exceptions=True,
                        )
                        yield cancelled
                        return
                if tool_task in done:
                    progress.close()
                    progress_task.cancel()
                    cancel_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    try:
                        await cancel_task
                    except asyncio.CancelledError:
                        pass
                    result = tool_task.result()
                    async for event in progress.drain():
                        yield self._event(event.type, event.data)
                    break
        except Exception as error:  # noqa: BLE001 - tool failures are user-visible events
            if not tool_task.done():
                tool_task.cancel()
            if not progress_task.done():
                progress_task.cancel()
            if not cancel_task.done():
                cancel_task.cancel()
            await asyncio.gather(
                tool_task,
                progress_task,
                cancel_task,
                return_exceptions=True,
            )
            await self._notify_tool_error(tool, call.arguments, error)
            self._stop_current_run = True
            message = f"{type(error).__name__}: {error}"
            self.messages.append(
                Message("tool", message, name=tool.name, tool_call_id=call.id)
            )
            yield self._tool_event(
                "tool_error",
                call,
                tool,
                {
                    "tool": tool.name,
                    "message": message,
                },
            )
            return
        except BaseException:
            if not tool_task.done():
                tool_task.cancel()
            if not progress_task.done():
                progress_task.cancel()
            if not cancel_task.done():
                cancel_task.cancel()
            progress.close()
            await asyncio.gather(
                tool_task,
                progress_task,
                cancel_task,
                return_exceptions=True,
            )
            raise

        try:
            text_result = await tool.text(result, **call.arguments)
        except Exception as error:  # noqa: BLE001 - formatter failures are tool errors
            await self._notify_tool_error(tool, call.arguments, error)
            self._stop_current_run = True
            message = f"{type(error).__name__}: {error}"
            self.messages.append(
                Message("tool", message, name=tool.name, tool_call_id=call.id)
            )
            yield self._tool_event(
                "tool_error",
                call,
                tool,
                {
                    "tool": tool.name,
                    "message": message,
                },
            )
            return
        else:
            self.messages.append(
                Message("tool", text_result, name=tool.name, tool_call_id=call.id)
            )
            await self._notify_after_tool(tool, call.arguments, result, text_result)
            yield self._tool_event(
                "tool_result",
                call,
                tool,
                {
                    "name": tool.name,
                    "result": result,
                    "text": text_result,
                },
            )

    def _cancel_event(self) -> Event | None:
        if not self._cancel_requested:
            return None
        self._stop_current_run = True
        self._run_status = "stopped"
        self._run_reason = self._cancel_reason
        return self._event("cancelled", {"reason": self._cancel_reason})

    def _find_tool(self, name: str, tools: list[Tool]) -> Tool | None:
        for tool in tools:
            if tool_matches_name(tool, name):
                return tool
        return None

    def _can_run_concurrently(self, calls: list[ToolCall], tools: list[Tool]) -> bool:
        if len(calls) < 2:
            return False
        for call in calls:
            tool = self._find_tool(call.name, tools)
            if tool is None or not tool.is_concurrency_safe(**call.arguments):
                return False
        return True

    async def _unknown_tool(self, call: ToolCall) -> AsyncGenerator[Event, None]:
        self._stop_current_run = True
        message = f"Unknown tool: {call.name}"
        self.messages.append(
            Message("tool", message, name=call.name, tool_call_id=call.id)
        )
        yield self._event("error", {"message": message})

    async def _invalid_tool_input(self, tool: Tool, call: ToolCall) -> Event | None:
        try:
            validation = await tool.validate(**call.arguments)
        except Exception as error:  # noqa: BLE001 - validation failures are user-visible
            validation_message = f"{type(error).__name__}: {error}"
            validation_ok = False
        else:
            validation_message = validation.message or f"Invalid input for {tool.name}"
            validation_ok = validation.ok

        if validation_ok:
            return None

        self._stop_current_run = True
        self.messages.append(
            Message("tool", validation_message, name=tool.name, tool_call_id=call.id)
        )
        return self._tool_event(
            "tool_invalid",
            call,
            tool,
            {
                "tool": tool.name,
                "message": validation_message,
            },
        )

    async def _check_permissions(
        self,
        tool: Tool,
        call: ToolCall,
    ) -> tuple[bool, list[Event]]:
        events: list[Event] = []
        permission_event_emitted = False
        for middleware in self.middleware:
            reason = self._permission_reason(middleware, tool, call.arguments)
            if reason and not permission_event_emitted:
                events.append(
                    self._tool_event(
                        "permission_request",
                        call,
                        tool,
                        {
                            "tool": tool.name,
                            "arguments": call.arguments,
                            "reason": reason,
                        },
                    )
                )
                permission_event_emitted = True

            hook = getattr(middleware, "before_tool_call", None)
            if hook is None:
                continue

            decision = await hook(tool, call.arguments)
            if decision.behavior == "allow":
                continue
            if decision.behavior == "ask":
                reason = reason or self._permission_reason(middleware, tool, call.arguments)
                if await self._ask_permission(
                    middleware,
                    tool,
                    call,
                    reason=reason,
                ):
                    continue
            message = decision.message or f"Permission denied for {tool.name}"
            self.messages.append(
                Message("tool", message, name=tool.name, tool_call_id=call.id)
            )
            events.append(
                self._tool_event(
                    "tool_denied",
                    call,
                    tool,
                    {
                        "tool": tool.name,
                        "message": message,
                    },
                )
            )
            return False, events
        return True, events

    async def _ask_permission(
        self,
        middleware: Middleware,
        tool: Tool,
        call: ToolCall,
        *,
        reason: str | None,
    ) -> bool:
        ask_user = getattr(middleware, "ask_user", None) or current_ask_user()
        if ask_user is None:
            return False

        request = PermissionRequest(
            tool=tool,
            arguments=call.arguments,
            reason=reason or "tool use",
        )
        if not await ask_allowed(ask_user, request):
            return False

        allow_request = getattr(middleware, "allow_request", None)
        if allow_request is not None:
            allow_request(request)
        return True

    async def _run_concurrent_tool(self, call: ToolCall, tool: Tool) -> dict:
        events: list[Event] = []
        progress = _Progress(tool.name)
        tool_task = asyncio.create_task(tool.run(**call.arguments, progress=progress))
        progress_task = asyncio.create_task(progress.next())

        try:
            while True:
                done, _ = await asyncio.wait(
                    {tool_task, progress_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if progress_task in done:
                    event = progress_task.result()
                    if event is not None:
                        events.append(event)
                    progress_task = asyncio.create_task(progress.next())
                if tool_task in done:
                    progress.close()
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass
                    result = tool_task.result()
                    async for event in progress.drain():
                        events.append(event)
                    text_result = await tool.text(result, **call.arguments)
                    return {"events": events, "result": result, "text": text_result}
        except asyncio.CancelledError:
            if not tool_task.done():
                tool_task.cancel()
            if not progress_task.done():
                progress_task.cancel()
            progress.close()
            await asyncio.gather(tool_task, progress_task, return_exceptions=True)
            raise
        except Exception as error:  # noqa: BLE001 - tool failures are user-visible
            if not tool_task.done():
                tool_task.cancel()
            if not progress_task.done():
                progress_task.cancel()
            progress.close()
            await asyncio.gather(
                tool_task,
                progress_task,
                return_exceptions=True,
            )
            await self._notify_tool_error(tool, call.arguments, error)
            return {"events": events, "error": error}

    @staticmethod
    def _permission_reason(middleware: Middleware, tool: Tool, arguments: dict) -> str | None:
        reason = getattr(middleware, "reason", None)
        if reason is not None:
            return reason(tool, arguments)
        if tool.is_read_only(**arguments):
            return None
        return getattr(tool.permission, "reason", None)

    async def _notify_after_tool(
        self,
        tool: Tool,
        arguments: dict,
        result,
        text: str,
    ) -> None:
        for middleware in self.middleware:
            hook = getattr(middleware, "after_tool_call", None)
            if hook is not None:
                await hook(tool, arguments, result, text)

    async def _notify_before_model(self, state: RunState) -> None:
        for middleware in self.middleware:
            hook = getattr(middleware, "before_model_call", None)
            if hook is not None:
                await hook(state)

    async def _notify_after_model(self, state: RunState) -> None:
        for middleware in self.middleware:
            hook = getattr(middleware, "after_model_call", None)
            if hook is not None:
                await hook(state)

    async def _notify_model_error(
        self,
        state: RunState,
        error: Exception,
    ) -> None:
        for middleware in self.middleware:
            hook = getattr(middleware, "on_model_error", None)
            if hook is not None:
                await hook(state, error)

    async def _notify_tool_error(
        self,
        tool: Tool,
        arguments: dict,
        error: Exception,
    ) -> None:
        for middleware in self.middleware:
            hook = getattr(middleware, "on_tool_error", None)
            if hook is not None:
                await hook(tool, arguments, error)

    def _state(
        self,
        step: int,
        active_tools: list[Tool],
        result,
    ) -> RunState:
        return RunState(
            step=step,
            max_steps=self.max_steps,
            messages=self.messages,
            active_tools=active_tools,
            result=result,
            run_id=getattr(self, "_run_id", None),
        )

    def _apply_decision(self, decision: RunDecision) -> None:
        self.messages.extend(decision.inject)

    def _event(
        self,
        type: str,
        data: dict,
        *,
        step: int | None = None,
    ) -> Event:
        payload = dict(data)
        if getattr(self, "_trace_events", False):
            payload.setdefault("run_id", self._run_id)
            active_step = step if step is not None else getattr(
                self,
                "_current_step",
                None,
            )
            if active_step is not None:
                payload.setdefault("step", active_step)
        return Event(type, payload)

    def _run_end_event(self) -> Event:
        return self._event(
            "run_end",
            {
                "agent": self.agent.name,
                "status": self._run_status,
                "reason": self._run_reason,
                "messages": len(self.messages),
            },
        )

    def _tool_event(
        self,
        type: str,
        call: ToolCall,
        tool: Tool,
        data: dict | None = None,
    ) -> Event:
        payload = dict(data) if data is not None else {
            "name": tool.name,
            "arguments": call.arguments,
        }
        if getattr(self, "_trace_events", False) and call.id:
            payload.setdefault("tool_call_id", call.id)
        return self._event(type, payload)


async def arun(
    agent: Agent,
    text: str,
    *,
    model: Model,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 20,
) -> str:
    runtime = Runtime(
        agent=agent,
        model=model,
        middleware=middleware,
        policy=policy,
        session=session,
        max_steps=max_steps,
    )
    return await runtime.ask(text, tools=tools)


def run(
    agent: Agent,
    text: str,
    *,
    model: Model,
    middleware: list[Middleware] | None = None,
    policy: RunPolicy | None = None,
    session: Session | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 20,
) -> str:
    return asyncio.run(
        arun(
            agent,
            text,
            model=model,
            middleware=middleware,
            policy=policy,
            session=session,
            tools=tools,
            max_steps=max_steps,
        )
    )


def _decision_status(decision: RunDecision) -> str:
    if decision.reason == "completed":
        return "completed"
    return "stopped"


class _Progress:
    def __init__(self, tool: str):
        self.tool = tool
        self._queue: asyncio.Queue[Event | None] = asyncio.Queue()

    async def __call__(self, message: str, **data) -> None:
        payload = {
            "tool": self.tool,
            "message": message,
        }
        if data:
            payload["data"] = data
        await self._queue.put(Event("tool_progress", payload))

    async def next(self) -> Event | None:
        return await self._queue.get()

    async def drain(self) -> AsyncGenerator[Event, None]:
        while not self._queue.empty():
            event = await self._queue.get()
            if event is not None:
                yield event

    def close(self) -> None:
        self._queue.put_nowait(None)
