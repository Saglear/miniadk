import asyncio

from miniadk import Message, model


async def main() -> None:
    result = await model().complete(
        [
            Message("system", "Reply with exactly miniadk-ok."),
            Message("user", "ping"),
        ],
        tools=[],
    )
    print(result.message)


if __name__ == "__main__":
    asyncio.run(main())
