import miniadk


def test_public_api_exports_are_real_and_public():
    names = list(miniadk.__all__)

    assert names
    assert len(names) == len(set(names))
    assert names == sorted(names, key=str.lower)
    assert all(not name.startswith("_") for name in names)
    assert all(hasattr(miniadk, name) for name in names)


def test_public_api_names_stay_short_and_pythonic():
    long_names = [
        name
        for name in miniadk.__all__
        if len(name.split("_")) > 3 or len(name) > 32
    ]

    assert long_names == []


def test_top_level_api_keeps_tiny_agent_path_available():
    @miniadk.tool
    def add(left: int, right: int) -> int:
        """Add two numbers."""
        return left + right

    agent = miniadk.Agent("calc", "Answer with math.", tools=[add])

    assert agent.name == "calc"
    assert add.name == "add"
    assert callable(miniadk.run_cli)
    assert callable(miniadk.make_tools)
    assert callable(miniadk.agentic)
    assert callable(miniadk.model)


def test_top_level_api_does_not_export_business_presets():
    assert "coder" not in miniadk.__all__
    assert not hasattr(miniadk, "coder")
