"""
Configuration entrypoints for generative text benchmark execution.

Defines parameter schemas and construction logic for creating benchmark runs from
scenario files or runtime arguments. Provides flexible configuration loading with
support for built-in scenarios, custom YAML/JSON files, and programmatic overrides.
Handles serialization of complex types including backends, processors, and profiles
for persistent storage and reproduction of benchmark configurations.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from guidellm.backends import BackendArgs
from guidellm.benchmark import Profile
from guidellm.data import DataArgs
from guidellm.schemas.random import RandomArgs

__all__ = [
    "BenchmarkGenerativeArgs",
    "BenchmarkGenerativeGlobalArgs",
]


class BenchmarkGenerativeGlobalArgs(BaseModel):
    backend: BackendArgs = Field(
        ...,
        description="Configuration for the backend to use for generation requests.",
    )
    profile: Profile = Field(
        description="Optional profile configuration for the benchmark execution.",
    )
    constraints: list[ConstraintArgs] = Field(
        default_factory=list,
        description="List of constraints to apply during benchmark execution.",
    )
    data_loader: DataLoaderArgs = Field(
        default=None,
        description="Optional data loader configuration for loading benchmark datasets.",
    )
    data_column_mapper: DataColumnMapperArgs = Field(
        default=None,
        description="Optional data column mapper configuration for mapping dataset columns to expected input formats.",
    )
    data_preprocessors: list[DataProcessorArgs] = Field(
        default_factory=list,
        description="List of data preprocessors to apply to the input data before generation.",
    )
    data_finalizers: DataFinalizerArgs = Field(
        description="Data finalizer configuration for processing generated outputs before evaluation.",
    )
    data: list[DataArgs] = Field(
        default_factory=list,
        description="List of data configurations for loading and processing benchmark datasets.",
    )
    seed: RandomArgs = Field(
        default_factory=RandomArgs,
        description="Random seed configuration for reproducibility of benchmark execution.",
    )


class BenchmarkOutputArgs(BaseModel):
    pass


class BenchmarkGenerativeArgs(BaseModel):
    global_: BenchmarkGenerativeGlobalArgs = Field(
        default_factory=BenchmarkGenerativeGlobalArgs,
        alias="global",
        description="Global configuration parameters for the generative benchmark execution.",
    )
    benchmarks: list[dict[str, Any] | None] = Field(
        default_factory=list,
        description="Individual benchmark overrides",
    )
    outputs: list[BenchmarkOutputArgs] = Field(
        default_factory=list,
        description="Benchmark outputs",
    )
