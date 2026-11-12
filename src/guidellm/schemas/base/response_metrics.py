"""Typed server-reported request metrics with extensible payload capture."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from guidellm.schemas.base.base import StandardBaseDict

__all__ = ["VLLMRequestMetrics", "VLLMSpeculativeDecodingMetrics"]

_Counter = Annotated[int, Field(strict=True, ge=0)]


class VLLMSpeculativeDecodingMetrics(StandardBaseDict):
    """vLLM speculative summary fields, retaining detailed arrays and future extras."""

    mean_acceptance_length: float | None = Field(
        default=None, description="Server-reported tokens per step including the bonus"
    )
    draft_acceptance_rate: float | None = Field(
        default=None, description="Server-reported accepted fraction of draft tokens"
    )
    num_spec_steps: _Counter | None = Field(
        default=None, description="Number of verification steps"
    )
    num_accepted_draft_tokens: _Counter | None = Field(
        default=None, description="Accepted draft tokens, excluding bonus tokens"
    )
    num_draft_tokens: _Counter | None = Field(
        default=None, description="Proposed draft tokens"
    )
    acceptance_histogram: list[_Counter] | None = Field(
        default=None,
        description="Index j counts steps accepting exactly j draft tokens",
    )


class VLLMRequestMetrics(StandardBaseDict):
    """vLLM response metrics; optional known fields support timing-only responses."""

    time_to_first_token_ms: float | None = Field(
        default=None, description="Server-side time to first token in milliseconds"
    )
    generation_time_ms: float | None = Field(
        default=None, description="Server-side generation time in milliseconds"
    )
    queue_time_ms: float | None = Field(
        default=None, description="Server-side queue time in milliseconds"
    )
    mean_itl_ms: float | None = Field(
        default=None, description="Server-side mean inter-token latency in milliseconds"
    )
    tokens_per_second: float | None = Field(
        default=None, description="Server-reported token generation rate"
    )
    speculative_decoding: VLLMSpeculativeDecodingMetrics | None = Field(
        default=None, description="Optional speculative decoding summary metrics"
    )
