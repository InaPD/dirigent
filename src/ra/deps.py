"""What the nodes are given. Tests build one of these with stubs."""

from dataclasses import dataclass
from typing import Any

from ra.config import Settings
from ra.store import RunStore


@dataclass(frozen=True)
class Deps:
    store: RunStore
    settings: Settings
    # Filled in later phases. Nodes that need one assert on it.
    llm: Any | None = None
    search: Any | None = None
