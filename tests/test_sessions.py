import pytest

from miniadk import (
    Compact,
    Message,
    ModelResult,
    ScriptedModel,
    Session,
    SessionStore,
    compact,
    sessions,
)


def test_session_store_loads_missing_name_as_empty_session(tmp_path):
    store = sessions(tmp_path)

    loaded = store.load("main")

    assert loaded == Session()
    assert store.exists("main") is False


def test_session_store_saves_and_loads_named_sessions(tmp_path):
    store = SessionStore(tmp_path)
    session = Session([Message("user", "hello")])

    store.save(session, "main")

    assert store.exists("main") is True
    assert store.load("main") == session
    assert store.names() == ["main"]


def test_session_store_encodes_names_as_single_files(tmp_path):
    store = sessions(tmp_path)
    session = Session([Message("user", "hello")])

    store.save(session, "../project/main")

    path = store.path("../project/main")
    assert path.parent == tmp_path
    assert path.name == "..%2Fproject%2Fmain.json"
    assert store.load("../project/main") == session
    assert store.names() == ["../project/main"]


def test_session_store_deletes_named_sessions(tmp_path):
    store = sessions(tmp_path)
    store.save(Session([Message("user", "hello")]), "main")

    assert store.delete("main") is True
    assert store.delete("main") is False
    assert store.names() == []


def test_session_store_rejects_empty_names(tmp_path):
    store = sessions(tmp_path)

    try:
        store.path("  ")
    except ValueError as error:
        assert str(error) == "session name cannot be empty"
    else:
        raise AssertionError("empty session names should be rejected")


async def test_compact_helper_summarizes_when_session_exceeds_threshold():
    session = Session(
        [
            Message("system", "Answer."),
            Message("user", "old"),
            Message("assistant", "old answer"),
            Message("user", "recent"),
        ]
    )
    model = ScriptedModel([ModelResult(message="Old summary.")])

    summary = await compact(session, model=model, spec=Compact(chars=1, keep=1))

    assert summary == "Old summary."
    assert session.messages == [
        Message("system", "Answer."),
        Message("system", "Old summary."),
        Message("user", "recent"),
    ]
    assert model.calls[0][0][1].content == "user: old\nassistant: old answer"


async def test_compact_helper_skips_when_session_is_under_threshold():
    session = Session([Message("system", "Answer."), Message("user", "short")])
    model = ScriptedModel([ModelResult(message="should not be used")])

    summary = await compact(session, model=model, spec=10_000)

    assert summary == ""
    assert len(model.calls) == 0


@pytest.mark.parametrize("spec", [False, None])
async def test_compact_helper_can_be_disabled(spec):
    session = Session([Message("system", "Answer."), Message("user", "long text")])
    model = ScriptedModel([ModelResult(message="should not be used")])

    summary = await compact(session, model=model, spec=spec)

    assert summary == ""
    assert len(model.calls) == 0
