from llm_judge import CORE_PROMPTS, FOLLOW_UP, PROBES


def test_prompts_loaded_from_yaml() -> None:
    assert CORE_PROMPTS, "Expected core prompts to be populated from YAML."
    assert PROBES, "Expected probes to be populated from YAML."
    assert isinstance(FOLLOW_UP, str) and FOLLOW_UP


def test_prompts_are_string_collections() -> None:
    assert all(isinstance(item, str) for item in CORE_PROMPTS)
    assert all(isinstance(item, str) for item in PROBES)
