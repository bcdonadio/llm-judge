from llm_judge import JUDGE_INSTRUCTIONS, JUDGE_SCHEMA, JUDGE_SYSTEM


def test_judge_config_loaded() -> None:
    assert isinstance(JUDGE_SYSTEM, str) and JUDGE_SYSTEM
    assert isinstance(JUDGE_INSTRUCTIONS, str) and JUDGE_INSTRUCTIONS
    assert isinstance(JUDGE_SCHEMA, dict) and JUDGE_SCHEMA


def test_instructions_include_schema() -> None:
    assert JUDGE_SCHEMA["required"]
    for field in JUDGE_SCHEMA["required"]:
        assert field in JUDGE_INSTRUCTIONS
