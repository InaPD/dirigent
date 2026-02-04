"""Prices. A wrong number here makes every cost in every trace quietly wrong."""

import pytest

from ra.config import ModelConfig
from ra.pricing import PRICES_USD_PER_MTOK, UnknownModel, cost_usd, price_for
from ra.schemas import Usage


def test_a_million_tokens_costs_the_headline_rate():
    cost = cost_usd("claude-sonnet-5", Usage(input_tokens=1_000_000))
    assert cost == pytest.approx(2.00)

    cost = cost_usd("claude-sonnet-5", Usage(output_tokens=1_000_000))
    assert cost == pytest.approx(10.00)


def test_haiku_is_the_cheaper_tier():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", usage) < cost_usd("claude-sonnet-5", usage)


def test_a_realistic_step_is_priced_to_the_cent():
    """1,200 in and 400 out on Haiku 4.5, at $1 and $5 per million."""
    cost = cost_usd("claude-haiku-4-5", Usage(input_tokens=1_200, output_tokens=400))
    assert cost == pytest.approx((1_200 * 1.00 + 400 * 5.00) / 1_000_000)


def test_cache_reads_are_cheaper_than_fresh_input():
    fresh = cost_usd("claude-sonnet-5", Usage(input_tokens=10_000))
    cached = cost_usd("claude-sonnet-5", Usage(cache_read_input_tokens=10_000))
    written = cost_usd("claude-sonnet-5", Usage(cache_creation_input_tokens=10_000))

    assert cached < fresh < written


def test_zero_usage_costs_nothing():
    assert cost_usd("claude-sonnet-5", Usage()) == 0.0


def test_an_unknown_model_says_what_to_do():
    with pytest.raises(UnknownModel, match="pricing.py"):
        price_for("claude-not-a-model")


def test_every_configured_model_has_a_price():
    """The guard that stops a run failing halfway through on a missing price."""
    models = ModelConfig()
    for role in ("planner", "reviewer", "researcher", "writer"):
        assert getattr(models, role) in PRICES_USD_PER_MTOK


def test_a_model_without_a_price_is_rejected_at_startup():
    with pytest.raises(ValueError, match="pricing.py"):
        ModelConfig(writer="claude-not-a-model")


@pytest.mark.parametrize("model", sorted(PRICES_USD_PER_MTOK))
def test_output_always_costs_more_than_input(model):
    p = PRICES_USD_PER_MTOK[model]
    assert p.output > p.input
    assert p.cache_read < p.input < p.cache_write
