"""Configuration: credential errors say what to do, and knobs read their env names."""

import pytest

from ra.config import MissingCredential, ModelConfig, Settings, get_settings


def test_missing_keys_are_not_an_import_time_failure():
    settings = Settings(anthropic_api_key=None, tavily_api_key=None)
    assert settings.anthropic_api_key is None


def test_require_anthropic_key_explains_the_fix():
    with pytest.raises(MissingCredential, match=r"ANTHROPIC_API_KEY.*\.env"):
        Settings(anthropic_api_key=None).require_anthropic_key()


def test_require_tavily_key_explains_the_fix():
    with pytest.raises(MissingCredential, match=r"TAVILY_API_KEY.*\.env"):
        Settings(tavily_api_key=None).require_tavily_key()


def test_keys_are_returned_when_present():
    settings = Settings(anthropic_api_key="sk-ant-x", tavily_api_key="tvly-y")
    assert settings.require_anthropic_key() == "sk-ant-x"
    assert settings.require_tavily_key() == "tvly-y"


def test_a_key_never_shows_up_in_a_repr():
    """SecretStr is what keeps a key out of logs and tracebacks."""
    settings = Settings(anthropic_api_key="sk-ant-supersecret")
    assert "supersecret" not in repr(settings)
    assert "supersecret" not in str(settings.anthropic_api_key)


def test_env_overrides_use_the_ra_prefix(monkeypatch):
    monkeypatch.setenv("RA_LEASE_TTL_S", "7")
    monkeypatch.setenv("RA_SWEEP_SECONDS", "3")
    monkeypatch.setenv("RA_STUB", "slow_research")
    settings = Settings()
    assert settings.lease_ttl_s == 7
    assert settings.sweeper_interval_s == 3
    assert settings.stub == "slow_research"


def test_worker_id_defaults_to_host_and_pid():
    assert ":" in Settings().worker_id


def test_model_ids_carry_no_date_suffix():
    """Date-suffixed ids are a stale habit and are rejected by the API."""
    models = ModelConfig()
    for name in ("planner", "reviewer", "researcher", "writer"):
        model_id = getattr(models, name)
        assert not model_id[-1].isdigit() or "-20" not in model_id


def test_get_settings_is_cached():
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()


def test_an_explicit_none_beats_a_key_in_the_environment(monkeypatch):
    """Why the test fixtures pin the keys: otherwise a developer's .env changes results."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-the-environment")

    assert Settings().anthropic_api_key is not None
    assert Settings(anthropic_api_key=None).anthropic_api_key is None
