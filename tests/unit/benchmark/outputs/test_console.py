"""Speculative decoding console table coverage."""

from __future__ import annotations

import pytest

from guidellm.benchmark.outputs.console import GenerativeBenchmarkerConsole
from guidellm.benchmark.schemas.metrics import SpeculativeDecodingSummary
from guidellm.scheduler import ConcurrentStrategy
from guidellm.schemas import GenerativeRequestStats
from tests.unit.benchmark.html_report_fixtures import make_benchmark, report


def _request(steps: int, accepted: int, drafted: int, histogram: list[int]):
    """Build request stats from server-reported speculative counters."""
    return GenerativeRequestStats.model_validate(
        {
            "request_id": "spec",
            "info": {"request_id": "spec"},
            "input_metrics": {},
            "output_metrics": {},
            "response_metrics": {
                "speculative_decoding": {
                    "num_spec_steps": steps,
                    "num_accepted_draft_tokens": accepted,
                    "num_draft_tokens": drafted,
                    "acceptance_histogram": histogram,
                }
            },
        }
    )


def _benchmark(summary: SpeculativeDecodingSummary | None, streams: int = 10):
    """Build a benchmark using the shared report fixtures."""
    benchmark = make_benchmark(
        strategy=ConcurrentStrategy(streams=streams), rps=1, tps=10
    )
    benchmark.metrics.speculative_decoding = summary
    return benchmark


def _phase_request(accepted: list[int], reasoning_tokens: int):
    """Build detailed counters with two drafts per step and reported phase usage."""
    request = _request(
        len(accepted),
        sum(accepted),
        2 * len(accepted),
        [accepted.count(position) for position in range(3)],
    )
    assert request.response_metrics is not None
    counters = request.response_metrics.speculative_decoding
    assert counters is not None
    counters.num_spec_tokens = 2
    counters.per_step_accepted = accepted
    counters.per_step_drafted = [2] * len(accepted)
    request.output_metrics.text_tokens = 1 + len(accepted) + sum(accepted)
    request.output_metrics.reasoning_tokens = reasoning_tokens
    return request


@pytest.mark.regression
@pytest.mark.asyncio
async def test_finalize_prints_summary_with_pooled_and_per_request_statistics(mocker):
    """Render completed-request statistics without mixing in other statuses.

    ## WRITTEN BY AI ##
    """
    outlier = _request(1, 2, 2, [0, 0, 1])
    summary = SpeculativeDecodingSummary.compile(
        [_request(2, 2, 4, [1, 0, 1]), _request(8, 0, 16, [8, 0, 0])],
        [outlier],
        [outlier],
    )
    measured = _benchmark(summary)
    other = _benchmark(summary, streams=20)
    output = GenerativeBenchmarkerConsole()
    mocker.patch.object(output.console, "print")
    table = mocker.patch.object(output.console, "print_table")

    assert (
        await output.finalize(report(measured, _benchmark(None), other))
        == "printed to console"
    )

    tables = {
        call.kwargs["title"]: call.args
        for call in table.call_args_list
        if "title" in call.kwargs
    }
    headers, values = tables["Speculative Decoding Summary (Completed Requests)"]
    assert headers == [
        ["Benchmark", "Strategy", ""],
        ["Benchmark", "Phase", ""],
        *[
            ["Acceptance Length", "tok/step", stat]
            for stat in ("Pooled", "Mean", "Mdn", "p95")
        ],
        *[
            ["Draft Acceptance Rate", "%", stat]
            for stat in ("Pooled", "Mean", "Mdn", "p95")
        ],
    ]
    assert values == [
        [str(measured.config.strategy), str(other.config.strategy)],
        ["Overall", "Overall"],
        *[
            [value, value]
            for value in (
                "1.20",
                "1.50",
                "1.00",
                "2.00",
                "10.00",
                "25.00",
                "0.00",
                "50.00",
            )
        ],
    ]
    titles = list(tables)
    summary_index = titles.index("Speculative Decoding Summary (Completed Requests)")
    assert titles[summary_index - 1] == "Request Token Statistics (Completed Requests)"
    assert (
        titles[summary_index + 1]
        == "Speculative Acceptance by Draft Position (Completed Requests)"
    )


@pytest.mark.regression
def test_position_table_orders_positions_and_groups_benchmarks(mocker):
    """Scale fractions only for display and keep multiple strategy blocks aligned.

    ## WRITTEN BY AI ##
    """
    summary = SpeculativeDecodingSummary.compile(
        [_request(2, 2, 4, [1, 0, 1]), _request(8, 0, 16, [8, 0, 0])], [], []
    )
    assert summary is not None
    summary.cumulative_acceptance = dict(
        reversed(list(summary.cumulative_acceptance.items()))
    )
    original = summary.model_dump()
    measured = _benchmark(summary)
    other = _benchmark(summary, streams=20)
    output = GenerativeBenchmarkerConsole()
    mocker.patch.object(output.console, "print")
    table = mocker.patch.object(output.console, "print_table")

    output.print_speculative_positions_table(report(measured, _benchmark(None), other))

    headers, values = table.call_args.args
    assert headers == [
        ["Benchmark", "Strategy", ""],
        ["Benchmark", "Phase", ""],
        ["Draft", "Position", ""],
        *[
            ["Cumulative Acceptance", "%", stat]
            for stat in ("Pooled", "Mean", "Mdn", "p95")
        ],
        *[
            ["Conditional Acceptance", "%", stat]
            for stat in ("Pooled", "Mean", "Mdn", "p95")
        ],
    ]
    assert values == [
        [str(measured.config.strategy), "", str(other.config.strategy), ""],
        ["Overall", "", "Overall", ""],
        ["1", "2", "1", "2"],
        ["10.00"] * 4,
        ["25.00"] * 4,
        ["0.00"] * 4,
        ["50.00"] * 4,
        ["10.00", "100.00"] * 2,
        ["25.00", "100.00"] * 2,
        ["0.00", "100.00"] * 2,
        ["50.00", "100.00"] * 2,
    ]
    assert summary.model_dump() == original


@pytest.mark.regression
def test_phase_tables_show_completed_statistics_and_group_mixed_benchmarks(mocker):
    """Render all four phase metrics and preserve overall-only benchmark alignment.

    ## WRITTEN BY AI ##
    """
    outlier = _phase_request([2, 2, 2], 4)
    summary = SpeculativeDecodingSummary.compile(
        [_phase_request([2, 0, 1], 4), _phase_request([0, 0, 0], 2)],
        [outlier],
        [outlier],
    )
    assert summary is not None
    for phase in (summary, summary.reasoning, summary.content):
        assert phase is not None
        phase.cumulative_acceptance = dict(
            reversed(list(phase.cumulative_acceptance.items()))
        )
    original = summary.model_dump()
    measured = _benchmark(summary)
    legacy = _benchmark(
        SpeculativeDecodingSummary.compile([_request(1, 0, 2, [1, 0, 0])], [], []),
        streams=20,
    )
    output = GenerativeBenchmarkerConsole()
    mocker.patch.object(output.console, "print")
    table = mocker.patch.object(output.console, "print_table")
    results = report(measured, _benchmark(None), legacy)

    output.print_speculative_decoding_table(results)
    output.print_speculative_positions_table(results)

    overall = table.call_args_list[0].args[1]
    assert overall == [
        [str(measured.config.strategy), "", "", str(legacy.config.strategy)],
        ["Overall", "Reasoning", "Output", "Overall"],
        ["1.50", "2.00", "1.25", "1.00"],
        ["1.50", "2.00", "1.25", "1.00"],
        ["1.00", "1.00", "1.00", "1.00"],
        ["2.00", "3.00", "1.50", "1.00"],
        ["25.00", "50.00", "12.50", "0.00"],
        ["25.00", "50.00", "12.50", "0.00"],
        ["0.00", "0.00", "0.00", "0.00"],
        ["50.00", "100.00", "25.00", "0.00"],
    ]
    positions = table.call_args_list[1].args[1]
    assert positions == [
        [
            str(measured.config.strategy),
            "",
            "",
            "",
            "",
            "",
            str(legacy.config.strategy),
            "",
        ],
        ["Overall", "", "Reasoning", "", "Output", "", "Overall", ""],
        ["1", "2"] * 4,
        ["33.33", "16.67", "50.00", "50.00", "25.00", "0.00", "0.00", "0.00"],
        ["33.33", "16.67", "50.00", "50.00", "25.00", "0.00", "0.00", "0.00"],
        ["0.00"] * 8,
        ["66.67", "33.33", "100.00", "100.00", "50.00", "0.00", "0.00", "0.00"],
        ["33.33", "50.00", "50.00", "100.00", "25.00", "0.00", "0.00", "--"],
        ["33.33", "50.00", "50.00", "100.00", "25.00", "0.00", "0.00", "--"],
        ["0.00", "50.00", "0.00", "100.00", "0.00", "0.00", "0.00", "--"],
        ["66.67", "50.00", "100.00", "100.00", "50.00", "0.00", "0.00", "--"],
    ]
    assert summary.model_dump() == original


@pytest.mark.regression
@pytest.mark.parametrize("completed", [True, False])
def test_empty_phase_and_uncompleted_phase_metrics_display_placeholders(
    mocker, completed
):
    """Keep empty reasoning and unavailable completed statistics undefined.

    ## WRITTEN BY AI ##
    """
    requests = [_phase_request([0, 0], 0)]
    summary = SpeculativeDecodingSummary.compile(
        requests if completed else [], [] if completed else requests, []
    )
    output = GenerativeBenchmarkerConsole()
    mocker.patch.object(output.console, "print")
    table = mocker.patch.object(output.console, "print_table")
    results = report(_benchmark(summary))

    output.print_speculative_decoding_table(results)
    output.print_speculative_positions_table(results)

    overall = table.call_args_list[0].args[1]
    assert overall[1] == ["Overall", "Reasoning", "Output"]
    assert overall[2:6] == [["1.00", "--", "1.00"] if completed else ["--"] * 3] * 4
    assert overall[6:10] == [["0.00", "--", "0.00"] if completed else ["--"] * 3] * 4
    positions = table.call_args_list[1].args[1]
    assert positions[1] == ["Overall", "", "Reasoning", "", "Output", ""]
    assert (
        positions[3:7]
        == [["0.00", "0.00", "--", "--", "0.00", "0.00"] if completed else ["--"] * 6]
        * 4
    )
    assert (
        positions[7:11]
        == [["0.00", "--", "--", "--", "0.00", "--"] if completed else ["--"] * 6] * 4
    )


@pytest.mark.regression
@pytest.mark.parametrize("completed", [True, False])
def test_zero_and_undefined_acceptance_are_distinct(mocker, completed):
    """Show measured zero, but use placeholders for undefined or empty samples.

    ## WRITTEN BY AI ##
    """
    requests = [_request(2, 0, 4, [2, 0, 0])]
    summary = SpeculativeDecodingSummary.compile(
        requests if completed else [], [] if completed else requests, []
    )
    output = GenerativeBenchmarkerConsole()
    mocker.patch.object(output.console, "print")
    table = mocker.patch.object(output.console, "print_table")
    results = report(_benchmark(summary))

    output.print_speculative_decoding_table(results)
    output.print_speculative_positions_table(results)

    overall = table.call_args_list[0].args[1]
    assert overall[2:6] == [["1.00" if completed else "--"]] * 4
    assert overall[6:10] == [["0.00" if completed else "--"]] * 4
    positions = table.call_args_list[1].args[1]
    assert positions[3:7] == [["0.00" if completed else "--"] * 2] * 4
    assert positions[7:11] == [["0.00" if completed else "--", "--"]] * 4


@pytest.mark.regression
@pytest.mark.parametrize("has_benchmark", [True, False])
def test_tables_omitted_without_speculative_metrics(mocker, has_benchmark):
    """Keep reports without speculative data free of empty tables and spacing.

    ## WRITTEN BY AI ##
    """
    output = GenerativeBenchmarkerConsole()
    print_ = mocker.patch.object(output.console, "print")
    table = mocker.patch.object(output.console, "print_table")
    results = report(*([_benchmark(None)] if has_benchmark else []))

    output.print_speculative_decoding_table(results)
    output.print_speculative_positions_table(results)

    table.assert_not_called()
    print_.assert_not_called()
