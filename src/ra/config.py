"""Process configuration, read once from the environment."""

import os
import socket
from functools import lru_cache

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Which Claude model backs each node.

    Verified against the Claude API model list on 14 Sep 2026. These ids are complete as
    written. Never append a date suffix to them.
    """

    planner: str = "claude-haiku-4-5"
    reviewer: str = "claude-haiku-4-5"
    researcher: str = "claude-sonnet-5"
    writer: str = "claude-sonnet-5"


def _default_worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


class MissingCredential(RuntimeError):
    """Raised when a node needs a key that was never configured."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Credentials are optional at import time. The API process never calls a model, and CI
    # runs without keys. Nodes that need one call require_* below and get a clear error.
    anthropic_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None

    redis_url: str = "redis://localhost:6379/0"

    lease_ttl_s: int = Field(
        default=60, validation_alias=AliasChoices("RA_LEASE_TTL_S", "lease_ttl_s")
    )
    sweeper_interval_s: int = Field(
        default=30, validation_alias=AliasChoices("RA_SWEEP_SECONDS", "sweeper_interval_s")
    )
    stub: str | None = Field(default=None, validation_alias=AliasChoices("RA_STUB", "stub"))
    worker_id: str = Field(
        default_factory=_default_worker_id,
        validation_alias=AliasChoices("RA_WORKER_ID", "worker_id"),
    )
    max_request_bytes: int = Field(
        default=8192, validation_alias=AliasChoices("RA_MAX_REQUEST_BYTES", "max_request_bytes")
    )
    rate_limit_per_min: int = Field(
        default=30, validation_alias=AliasChoices("RA_RATE_LIMIT_PER_MIN", "rate_limit_per_min")
    )

    models: ModelConfig = ModelConfig()

    def require_anthropic_key(self) -> str:
        if self.anthropic_api_key is None:
            raise MissingCredential("ANTHROPIC_API_KEY is not set. Copy .env.example to .env.")
        return self.anthropic_api_key.get_secret_value()

    def require_tavily_key(self) -> str:
        if self.tavily_api_key is None:
            raise MissingCredential("TAVILY_API_KEY is not set. Copy .env.example to .env.")
        return self.tavily_api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
