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
    assert overall[1:5] == [["1.00" if completed else "--"]] * 4
    assert overall[5:9] == [["0.00" if completed else "--"]] * 4
    positions = table.call_args_list[1].args[1]
    assert positions[2:6] == [["0.00" if completed else "--"] * 2] * 4
    assert positions[6:10] == [["0.00" if completed else "--", "--"]] * 4


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
