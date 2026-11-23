"""Tests for interpreting optional vLLM speculative decoding payloads."""

from __future__ import annotations

from typing import Any

import pytest

from guidellm.schemas import SpeculativeDecodingMetrics, VLLMRequestMetrics


def _interpret(metrics: dict[str, Any] | None) -> SpeculativeDecodingMetrics | None:
    """Interpret an already validated response payload."""
    return SpeculativeDecodingMetrics.from_response_metrics(
        VLLMRequestMetrics.model_validate(metrics) if metrics is not None else None
    )


@pytest.mark.smoke
def test_vllm_histogram_and_derived_values():
    """Check exact-length histogram semantics against the upstream example.

    ## WRITTEN BY AI ##
    """
    raw = {
        "num_spec_steps": 43,
        "num_accepted_draft_tokens": 10,
        "num_draft_tokens": 129,
        "acceptance_histogram": [39, 1, 0, 3],
        "mean_acceptance_length": 999,
        "draft_acceptance_rate": 999,
    }
    metrics = _interpret({"speculative_decoding": raw})
    assert metrics is not None
    assert metrics.acceptance_length == pytest.approx(1 + 10 / 43)
    assert metrics.acceptance_rate == pytest.approx(1000 / 129)
    assert metrics.acceptance_tails == [4, 3, 3]
    assert metrics.cumulative_acceptance == pytest.approx(
        {1: 4 / 43, 2: 3 / 43, 3: 3 / 43}
    )
    assert metrics.conditional_acceptance == pytest.approx({1: 4 / 43, 2: 0.75, 3: 1})
    assert metrics.position_counts(4, conditional=False) == (0, 43)
    assert metrics.position_counts(4, conditional=True) == (0, 3)
    assert metrics.position_counts(5, conditional=True) == (0, 0)
    assert raw["acceptance_histogram"] == [39, 1, 0, 3]
    restored = SpeculativeDecodingMetrics.model_validate_json(metrics.model_dump_json())
    assert restored.cumulative_acceptance == metrics.cumulative_acceptance


@pytest.mark.regression
@pytest.mark.parametrize(
    "raw", [None, {}, {"speculative_decoding": {}}, {"speculative_decoding": None}]
)
def test_missing_payload(raw):
    """Unsupported and absent optional payloads produce no derived values.

    ## WRITTEN BY AI ##
    """
    assert _interpret(raw) is None


@pytest.mark.regression
@pytest.mark.parametrize("histogram", [[], [1, 0], [0, 2]])
def test_invalid_histogram_preserves_scalar_metrics(histogram):
    """Reject malformed or inconsistent histograms without losing scalar ratios.

    ## WRITTEN BY AI ##
    """
    metrics = _interpret(
        {
            "speculative_decoding": {
                "num_spec_steps": 2,
                "num_accepted_draft_tokens": 0,
                "num_draft_tokens": 4,
                "acceptance_histogram": histogram,
            }
        }
    )
    assert metrics is not None
    assert metrics.acceptance_length == 1
    assert metrics.acceptance_rate == 0
    assert metrics.acceptance_histogram is None


@pytest.mark.sanity
@pytest.mark.parametrize("steps", [0, 2])
def test_zero_denominators_and_unreachable_positions(steps):
    """Zero acceptance is measurable; empty denominators remain undefined.

    ## WRITTEN BY AI ##
    """
    metrics = _interpret(
        {
            "speculative_decoding": {
                "num_spec_steps": steps,
                "num_accepted_draft_tokens": 0,
                "num_draft_tokens": 0,
                "acceptance_histogram": [steps, 0, 0],
            }
        }
    )
    assert metrics is not None
    assert metrics.acceptance_length == (1 if steps else None)
    assert metrics.acceptance_rate is None
    assert metrics.cumulative_acceptance == {
        1: 0 if steps else None,
        2: 0 if steps else None,
    }
    assert metrics.conditional_acceptance == {1: 0 if steps else None, 2: None}


@pytest.mark.regression
@pytest.mark.parametrize(("steps", "drafted"), [(0, 4), (2, 1)])
def test_inconsistent_accepted_counter(steps, drafted):
    """Impossible accepted counts do not become valid scalar measurements.

    ## WRITTEN BY AI ##
    """
    metrics = _interpret(
        {
            "speculative_decoding": {
                "num_spec_steps": steps,
                "num_accepted_draft_tokens": 2,
                "num_draft_tokens": drafted,
            }
        }
    )
    assert metrics is not None
    assert metrics.num_accepted_draft_tokens is None
    assert metrics.acceptance_length is None
    assert metrics.acceptance_rate is None


def _detailed_metrics(**overrides: Any) -> VLLMRequestMetrics:
    """Four steps at reconstructed zero-based starts 1, 4, 5, and 7."""
    return VLLMRequestMetrics.model_validate(
        {
            "speculative_decoding": {
                "num_spec_steps": 4,
                "num_accepted_draft_tokens": 6,
                "num_draft_tokens": 9,
                "num_spec_tokens": 3,
                "acceptance_histogram": [1, 1, 1, 1],
                "per_step_accepted": [2, 0, 1, 3],
                "per_step_drafted": [3, 1, 2, 3],
                **overrides,
            }
        }
    )


@pytest.mark.regression
def test_phase_acceptance_and_round_trip():
    """Keep a crossing step and its target token entirely in its starting phase.

    ## WRITTEN BY AI ##
    """
    raw = _detailed_metrics()
    before = raw.model_dump()
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
        raw, reasoning_tokens=6, output_tokens=11
    )
    assert metrics is not None
    reasoning, content = metrics.reasoning, metrics.content
    assert reasoning is not None
    assert content is not None
    assert reasoning.num_spec_steps == 3
    assert reasoning.num_accepted_draft_tokens == 3
    assert reasoning.num_draft_tokens == 6
    assert reasoning.acceptance_histogram == [1, 1, 1, 0]
    assert reasoning.acceptance_length == 2
    assert reasoning.acceptance_rate == 50
    assert reasoning.cumulative_acceptance == {1: 2 / 3, 2: 1 / 3, 3: 0}
    assert reasoning.conditional_acceptance == {1: 2 / 3, 2: 0.5, 3: 0}
    assert content.num_spec_steps == 1
    assert content.num_accepted_draft_tokens == 3
    assert content.num_draft_tokens == 3
    assert content.acceptance_length == 4
    assert content.acceptance_rate == 100
    assert content.cumulative_acceptance == {1: 1, 2: 1, 3: 1}
    assert content.conditional_acceptance == {1: 1, 2: 1, 3: 1}
    assert metrics.num_spec_steps == reasoning.num_spec_steps + content.num_spec_steps
    assert metrics.num_accepted_draft_tokens == (
        reasoning.num_accepted_draft_tokens + content.num_accepted_draft_tokens
    )
    assert (
        metrics.num_draft_tokens
        == reasoning.num_draft_tokens + content.num_draft_tokens
    )
    assert raw.model_dump() == before
    restored = SpeculativeDecodingMetrics.model_validate_json(metrics.model_dump_json())
    assert restored == metrics


@pytest.mark.regression
@pytest.mark.parametrize(
    ("reasoning_tokens", "reasoning_steps"),
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 2), (6, 3), (7, 3), (11, 4)],
)
def test_phase_boundaries(reasoning_tokens, reasoning_steps):
    """Account for prefill and test boundaries inside drafts, at bonuses, and steps.

    ## WRITTEN BY AI ##
    """
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
        _detailed_metrics(), reasoning_tokens=reasoning_tokens, output_tokens=11
    )
    assert metrics.reasoning.num_spec_steps == reasoning_steps
    assert metrics.content.num_spec_steps == 4 - reasoning_steps
    if reasoning_steps == 0:
        assert metrics.reasoning.acceptance_length is None
        assert metrics.reasoning.acceptance_rate is None
        assert metrics.reasoning.cumulative_acceptance == {1: None, 2: None, 3: None}
        assert metrics.reasoning.conditional_acceptance == {1: None, 2: None, 3: None}
    if reasoning_steps == 4:
        assert metrics.content.acceptance_length is None


@pytest.mark.regression
def test_phase_terminal_truncation_and_optional_summary():
    """Terminal trimming does not change step attribution or reported acceptance.

    ## WRITTEN BY AI ##
    """
    full = SpeculativeDecodingMetrics.from_response_metrics(
        _detailed_metrics(), reasoning_tokens=6, output_tokens=11
    )
    truncated = SpeculativeDecodingMetrics.from_response_metrics(
        _detailed_metrics(), reasoning_tokens=6, output_tokens=9
    )
    assert truncated == full
    details_only = SpeculativeDecodingMetrics.from_response_metrics(
        _detailed_metrics(
            num_spec_steps=None,
            num_accepted_draft_tokens=None,
            num_draft_tokens=None,
            acceptance_histogram=None,
            num_spec_tokens=None,
        ),
        reasoning_tokens=6,
        output_tokens=11,
    )
    assert details_only.acceptance_length is None
    assert details_only.reasoning == full.reasoning
    assert details_only.content == full.content


@pytest.mark.regression
@pytest.mark.parametrize(
    "overrides",
    [
        {"per_step_accepted": None},
        {"per_step_drafted": None},
        {"per_step_accepted": [2, 0, 1]},
        {"per_step_accepted": [2, 0, 1, 4]},
        {"per_step_drafted": [3, 0, 2, 3]},
        {"per_step_accepted": [1, 1, 1, 3]},
        {"num_spec_steps": 5},
        {"num_accepted_draft_tokens": 5},
        {"num_draft_tokens": 10},
        {"num_spec_tokens": 2},
        {"acceptance_histogram": []},
        {"acceptance_histogram": [0, 3, 0, 1]},
    ],
)
def test_inconsistent_details_preserve_overall_metrics(overrides):
    """Invalid detailed data cannot create phase ratios or invalidate overall ratios.

    ## WRITTEN BY AI ##
    """
    raw = _detailed_metrics(**overrides)
    overall = SpeculativeDecodingMetrics.from_response_metrics(raw)
    phased = SpeculativeDecodingMetrics.from_response_metrics(
        raw, reasoning_tokens=6, output_tokens=11
    )
    assert phased == overall
    assert phased.reasoning is None
    assert phased.content is None
    assert phased.acceptance_length is not None
    assert phased.acceptance_rate is not None


@pytest.mark.regression
@pytest.mark.parametrize(
    ("reasoning", "output"), [(None, 11), (6, None), (-1, 11), (12, 11)]
)
def test_unavailable_phase_usage(reasoning, output):
    """Missing or inconsistent usage never becomes an inferred zero reasoning count.

    ## WRITTEN BY AI ##
    """
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
        _detailed_metrics(), reasoning_tokens=reasoning, output_tokens=output
    )
    assert metrics.reasoning is None
    assert metrics.content is None
    assert metrics.acceptance_length == 2.5


@pytest.mark.sanity
@pytest.mark.parametrize("steps", [0, 2])
def test_phase_zero_acceptance(steps):
    """Empty steps are undefined while zero accepted drafts remain measured zeros.

    ## WRITTEN BY AI ##
    """
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
        _detailed_metrics(
            num_spec_steps=steps,
            num_accepted_draft_tokens=0,
            num_draft_tokens=steps,
            acceptance_histogram=[steps, 0, 0, 0],
            per_step_accepted=[0] * steps,
            per_step_drafted=[1] * steps,
        ),
        reasoning_tokens=0,
        output_tokens=steps + 1,
    )
    assert metrics.content.acceptance_rate == (0 if steps else None)
    assert metrics.content.acceptance_length == (1 if steps else None)
    assert metrics.content.cumulative_acceptance[1] == (0 if steps else None)
    assert metrics.content.conditional_acceptance[2] is None
