import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

import yaml
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from guidellm.benchmark.benchmark import GenerativeBenchmark
from guidellm.benchmark.profile import (
    AsyncProfile,
    ConcurrentProfile,
    SweepProfile,
    ThroughputProfile,
)
from guidellm.objects import StandardBaseModel
from guidellm.scheduler import strategy_display_str
from guidellm.utils import Colors

__all__ = [
    "GenerativeBenchmarksReport",
    "save_generative_benchmarks",
    "GenerativeBenchmarksConsole",
]


class GenerativeBenchmarksReport(StandardBaseModel):
    benchmarks: List[GenerativeBenchmark]

    def save_file(self, path: Path):
        if path.is_dir():
            path = path / "benchmarks.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        extension = path.suffix.lower()

        if extension == ".json":
            self.save_json(path)
        elif extension in [".yaml", ".yml"]:
            self.save_yaml(path)
        elif extension in [".csv"]:
            self.save_csv(path)
        else:
            raise ValueError(f"Unsupported file extension: {extension} for {path}.")

    def save_json(self, path: Path):
        model_dict = self.model_dump()
        model_json = json.dumps(model_dict)

        with path.open("w") as file:
            file.write(model_json)

    def save_yaml(self, path: Path):
        model_dict = self.model_dump()
        model_yaml = yaml.dump(model_dict)

        with path.open("w") as file:
            file.write(model_yaml)

    def save_csv(self, path: Path):
        raise NotImplementedError("CSV format is not implemented yet.")


def save_generative_benchmarks(
    benchmarks: List[GenerativeBenchmark], path: Union[Path, str]
):
    path = Path(path) if isinstance(path, str) else path
    report = GenerativeBenchmarksReport(benchmarks=benchmarks)
    report.save_file(path)


class GenerativeBenchmarksConsole:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.benchmarks: Optional[List[GenerativeBenchmark]] = None
        self.console = Console()

    @property
    def benchmarks_profile_str(self) -> str:
        profile = self.benchmarks[0].args.profile if self.benchmarks else None

        if profile is None:
            return "None"

        profile_args = OrderedDict(
            {
                "type": profile.type_,
                "strategies": profile.strategy_types,
            }
        )

        if isinstance(profile, ConcurrentProfile):
            profile_args["streams"] = str(profile.streams)
        elif isinstance(profile, ThroughputProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
        elif isinstance(profile, AsyncProfile):
            profile_args["max_concurrency"] = str(profile.max_concurrency)
            profile_args["rate"] = str(profile.rate)
            profile_args["initial_burst"] = str(profile.initial_burst)
        elif isinstance(profile, SweepProfile):
            profile_args["sweep_size"] = str(profile.sweep_size)

        return ", ".join(f"{key}={value}" for key, value in profile_args.items())

    @property
    def benchmarks_args_str(self) -> str:
        args = self.benchmarks[0].args if self.benchmarks else None

        if args is None:
            return "None"

        args_dict = OrderedDict(
            {
                "max_number": args.max_number,
                "max_duration": args.max_duration,
                "warmup_number": args.warmup_number,
                "warmup_duration": args.warmup_duration,
                "cooldown_number": args.cooldown_number,
                "cooldown_duration": args.cooldown_duration,
            }
        )

        return ", ".join(f"{key}={value}" for key, value in args_dict.items())

    @property
    def benchmarks_worker_desc_str(self) -> str:
        return str(self.benchmarks[0].worker) if self.benchmarks else "None"

    @property
    def benchmarks_request_loader_desc_str(self) -> str:
        return str(self.benchmarks[0].request_loader) if self.benchmarks else "None"

    @property
    def benchmarks_extras_str(self) -> str:
        extras = self.benchmarks[0].extras if self.benchmarks else None

        if not extras:
            return "None"

        return ", ".join(f"{key}={value}" for key, value in extras.items())

    def print_section_header(self, title: str, new_lines: int = 2):
        if not self.enabled:
            return

        text = Text()

        for _ in range(new_lines):
            text.append("\n")

        text.append(f"{title}:", style=f"bold underline {Colors.INFO}")
        self.console.print(text)

    def print_labeled_line(self, label: str, value: str, indent: int = 4):
        if not self.enabled:
            return

        text = Text()
        text.append(label + ": ", style=f"bold {Colors.INFO}")
        text.append(": ")
        text.append(value, style="italic")
        self.console.print(
            Padding.indent(text, indent),
        )

    def print_line(self, value: str, indent: int = 0):
        if not self.enabled:
            return

        text = Text(value)
        self.console.print(
            Padding.indent(text, indent),
        )

    def print_table(self, headers: List[str], rows: List[List[Any]], title: str):
        if not self.enabled:
            return

        self.print_section_header(title)
        table = Table(*headers, header_style=f"bold {Colors.INFO}")

        for row in rows:
            table.add_row(*[Text(item, style="italic") for item in row])

        self.console.print(table)

    def print_benchmarks_metadata(self):
        if not self.enabled:
            return

        if not self.benchmarks:
            raise ValueError(
                "No benchmarks to print metadata for. Please set benchmarks first."
            )

        start_time = self.benchmarks[0].run_stats.start_time
        end_time = self.benchmarks[0].run_stats.end_time
        duration = end_time - start_time

        self.print_section_header("Benchmarks Completed")
        self.print_labeled_line("Run id", str(self.benchmarks[0].run_id))
        self.print_labeled_line(
            "Duration",
            f"{duration:.1f} seconds",
        )
        self.print_labeled_line(
            "Profile",
            self.benchmarks_profile_str,
        )
        self.print_labeled_line(
            "Args",
            self.benchmarks_args_str,
        )
        self.print_labeled_line(
            "Worker",
            self.benchmarks_worker_desc_str,
        )
        self.print_labeled_line(
            "Request Loader",
            self.benchmarks_request_loader_desc_str,
        )
        self.print_labeled_line(
            "Extras",
            self.benchmarks_extras_str,
        )

    def print_benchmarks_info(self):
        if not self.enabled:
            return

        if not self.benchmarks:
            raise ValueError(
                "No benchmarks to print info for. Please set benchmarks first."
            )

        headers = [
            "Benchmark",
            "Start Time",
            "End Time",
            "Duration (sec)",
            "Requests Made \n(comp / inc / err)",
            "Prompt Tok / Req \n(comp / inc / err)",
            "Output Tok / Req \n(comp / inc / err)",
            "Prompt Tok Total \n(comp / inc / err)",
            "Output Tok Total \n(comp / inc / err)",
        ]
        rows = []

        for benchmark in self.benchmarks:
            rows.append(
                [
                    strategy_display_str(benchmark.args.strategy),
                    f"{datetime.fromtimestamp(benchmark.start_time).strftime('%H:%M:%S')}",
                    f"{datetime.fromtimestamp(benchmark.end_time).strftime('%H:%M:%S')}",
                    f"{(benchmark.end_time - benchmark.start_time):.1f}",
                    (
                        f"{benchmark.request_totals.successful:>5} / "
                        f"{benchmark.request_totals.incomplete} / "
                        f"{benchmark.request_totals.errored}"
                    ),
                    (
                        f"{benchmark.metrics.prompt_token_count.successful.mean:>5.1f} / "  # noqa: E501
                        f"{benchmark.metrics.prompt_token_count.incomplete.mean:.1f} / "
                        f"{benchmark.metrics.prompt_token_count.errored.mean:.1f}"
                    ),
                    (
                        f"{benchmark.metrics.output_token_count.successful.mean:>5.1f} / "  # noqa: E501
                        f"{benchmark.metrics.output_token_count.incomplete.mean:.1f} / "
                        f"{benchmark.metrics.output_token_count.errored.mean:.1f}"
                    ),
                    (
                        f"{benchmark.metrics.prompt_token_count.successful.total_sum:>6.0f} / "  # noqa: E501
                        f"{benchmark.metrics.prompt_token_count.incomplete.total_sum:.0f} / "  # noqa: E501
                        f"{benchmark.metrics.prompt_token_count.errored.total_sum:.0f}"
                    ),
                    (
                        f"{benchmark.metrics.output_token_count.successful.total_sum:>6.0f} / "  # noqa: E501
                        f"{benchmark.metrics.output_token_count.incomplete.total_sum:.0f} / "  # noqa: E501
                        f"{benchmark.metrics.output_token_count.errored.total_sum:.0f}"
                    ),
                ]
            )

        self.print_table(headers=headers, rows=rows, title="Benchmarks Info")

    def print_benchmarks_stats(self):
        if not self.enabled:
            return

        if not self.benchmarks:
            raise ValueError(
                "No benchmarks to print stats for. Please set benchmarks first."
            )

        headers = [
            "Benchmark",
            "Requests / sec",
            "Requests Concurrency",
            "Output Tok / sec",
            "Total Tok / sec",
            "Req Latency (ms)\n(mean / median / p99)",
            "TTFT (ms)\n(mean / median / p99)",
            "ITL (ms)\n(mean / median / p99)",
            "TPOT (ms)\n(mean / median / p99)",
        ]
        rows = []

        for benchmark in self.benchmarks:
            rows.append(
                [
                    strategy_display_str(benchmark.args.strategy),
                    f"{benchmark.metrics.requests_per_second.successful.mean:.2f}",
                    f"{benchmark.metrics.request_concurrency.successful.mean:.2f}",
                    f"{benchmark.metrics.output_tokens_per_second.total.mean:.1f}",
                    f"{benchmark.metrics.tokens_per_second.total.mean:.1f}",
                    (
                        f"{benchmark.metrics.request_latency.successful.mean:.2f} / "
                        f"{benchmark.metrics.request_latency.successful.median:.2f} / "
                        f"{benchmark.metrics.request_latency.successful.percentiles.p99:.2f}"
                    ),
                    (
                        f"{benchmark.metrics.time_to_first_token_ms.successful.mean:.1f} / "  # noqa: E501
                        f"{benchmark.metrics.time_to_first_token_ms.successful.median:.1f} / "  # noqa: E501
                        f"{benchmark.metrics.time_to_first_token_ms.successful.percentiles.p99:.1f}"
                    ),
                    (
                        f"{benchmark.metrics.inter_token_latency_ms.successful.mean:.1f} / "  # noqa: E501
                        f"{benchmark.metrics.inter_token_latency_ms.successful.median:.1f} / "  # noqa: E501
                        f"{benchmark.metrics.inter_token_latency_ms.successful.percentiles.p99:.1f}"
                    ),
                    (
                        f"{benchmark.metrics.time_per_output_token_ms.successful.mean:.1f} / "  # noqa: E501
                        f"{benchmark.metrics.time_per_output_token_ms.successful.median:.1f} / "  # noqa: E501
                        f"{benchmark.metrics.time_per_output_token_ms.successful.percentiles.p99:.1f}"
                    ),
                ]
            )

        self.print_table(
            headers=headers,
            rows=rows,
            title="Benchmarks Stats",
        )
