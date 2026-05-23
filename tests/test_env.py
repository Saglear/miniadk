import os

from miniadk import load_env, load_env_upwards


def test_load_env_reads_simple_key_value_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MINIADK_TEST_A=hello\nMINIADK_TEST_B='quoted value'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MINIADK_TEST_A", raising=False)
    monkeypatch.delenv("MINIADK_TEST_B", raising=False)

    loaded = load_env(env_file)

    assert loaded == {"MINIADK_TEST_A": "hello", "MINIADK_TEST_B": "quoted value"}
    assert os.environ["MINIADK_TEST_A"] == "hello"
    assert os.environ["MINIADK_TEST_B"] == "quoted value"


def test_load_env_does_not_override_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MINIADK_TEST_A=file\n", encoding="utf-8")
    monkeypatch.setenv("MINIADK_TEST_A", "existing")

    loaded = load_env(env_file)

    assert loaded == {}
    assert os.environ["MINIADK_TEST_A"] == "existing"


def test_load_env_accepts_export_prefix_and_inline_comments(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "export MINIADK_TEST_A=hello # comment",
                "MINIADK_TEST_B=\"quoted # value\"",
                "MINIADK_TEST_C='single # value'",
            ]
        ),
        encoding="utf-8",
    )
    for name in ["MINIADK_TEST_A", "MINIADK_TEST_B", "MINIADK_TEST_C"]:
        monkeypatch.delenv(name, raising=False)

    loaded = load_env(env_file)

    assert loaded == {
        "MINIADK_TEST_A": "hello",
        "MINIADK_TEST_B": "quoted # value",
        "MINIADK_TEST_C": "single # value",
    }


def test_load_env_upwards_can_start_from_file_path(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    nested = tmp_path / "src" / "app.py"
    nested.parent.mkdir()
    nested.write_text("", encoding="utf-8")
    env_file.write_text("MINIADK_TEST_A=upwards\n", encoding="utf-8")
    monkeypatch.delenv("MINIADK_TEST_A", raising=False)

    found = load_env_upwards(start=nested)

    assert found == env_file
    assert os.environ["MINIADK_TEST_A"] == "upwards"
