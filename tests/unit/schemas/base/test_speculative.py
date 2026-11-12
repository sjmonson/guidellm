"""Tests for interpreting optional vLLM speculative decoding payloads."""

from __future__ import annotations

from typing import Any

import pytest

from guidellm.schemas import SpeculativeDecodingMetrics


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
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
        {"speculative_decoding": raw}
    )
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
    assert SpeculativeDecodingMetrics.from_response_metrics(raw) is None


@pytest.mark.regression
@pytest.mark.parametrize("bad", [True, -1, 1.5, "2", float("nan"), float("inf"), []])
def test_invalid_counters_do_not_coerce(bad: Any):
    """Malformed counters are unavailable while independent valid fields survive.

    ## WRITTEN BY AI ##
    """
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
        {
            "speculative_decoding": {
                "num_spec_steps": bad,
                "num_accepted_draft_tokens": 0,
                "num_draft_tokens": 4,
                "acceptance_histogram": [2, 0],
            }
        }
    )
    assert metrics is not None
    assert metrics.acceptance_length is None
    assert metrics.acceptance_rate == 0
    assert metrics.acceptance_tails is None


@pytest.mark.regression
@pytest.mark.parametrize(
    "histogram", [[], [1, 0], [0, 2], [2, -1, 1], [True, 1], "bad"]
)
def test_invalid_histogram_preserves_scalar_metrics(histogram):
    """Reject malformed or inconsistent histograms without losing scalar ratios.

    ## WRITTEN BY AI ##
    """
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
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
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
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
    metrics = SpeculativeDecodingMetrics.from_response_metrics(
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
