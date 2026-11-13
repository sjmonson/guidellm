"""Interpret vLLM per-request speculative-decoding counters."""

from __future__ import annotations

from functools import cached_property

from pydantic import Field, computed_field

from guidellm.schemas.base.base import StandardBaseModel
from guidellm.schemas.base.response_metrics import (
    VLLMRequestMetrics,
    VLLMSpeculativeDecodingMetrics,
)

__all__ = ["SpeculativeAcceptanceMetrics", "SpeculativeDecodingMetrics"]


class SpeculativeAcceptanceMetrics(StandardBaseModel):
    """Validated counters and derived values, excluding the bonus token from drafts."""

    num_spec_steps: int | None = None
    num_accepted_draft_tokens: int | None = None
    num_draft_tokens: int | None = None
    acceptance_histogram: list[int] | None = None

    @cached_property
    def acceptance_tails(self) -> list[int] | None:
        """:return: Counts accepting at least positions 1 onward, or None."""
        if self.acceptance_histogram is None:
            return None
        tails = [0] * (len(self.acceptance_histogram) - 1)
        tail = 0
        for j in range(len(self.acceptance_histogram) - 1, 0, -1):
            tail += self.acceptance_histogram[j]
            tails[j - 1] = tail
        return tails

    @computed_field  # type: ignore[misc]
    @property
    def cumulative_acceptance(self) -> dict[int, float | None]:
        """:return: Fractions accepting at least each one-based draft position."""
        return self._position_ratios(conditional=False)

    @computed_field  # type: ignore[misc]
    @property
    def conditional_acceptance(self) -> dict[int, float | None]:
        """:return: Acceptance fractions conditional on the preceding position."""
        return self._position_ratios(conditional=True)

    def _position_ratios(self, *, conditional: bool) -> dict[int, float | None]:
        return {
            position: numerator / denominator if denominator else None
            for position in range(1, len(self.acceptance_tails or []) + 1)
            for numerator, denominator in [
                self.position_counts(position, conditional=conditional)
            ]
        }

    @computed_field  # type: ignore[misc]
    @property
    def acceptance_length(self) -> float | None:
        """:return: Tokens per verification step including the bonus, or None."""
        if not self.num_spec_steps or self.num_accepted_draft_tokens is None:
            return None
        return 1 + self.num_accepted_draft_tokens / self.num_spec_steps

    @computed_field  # type: ignore[misc]
    @property
    def acceptance_rate(self) -> float | None:
        """:return: Percentage of proposed draft tokens accepted, or None."""
        if not self.num_draft_tokens or self.num_accepted_draft_tokens is None:
            return None
        return 100 * self.num_accepted_draft_tokens / self.num_draft_tokens

    def position_counts(self, position: int, *, conditional: bool) -> tuple[int, int]:
        """
        Return numerator and denominator for a one-based draft position.

        Positions beyond the histogram have zero accepting steps.

        :param position: Draft position, starting at one
        :param conditional: Divide by previous tail instead of all steps
        :return: Accepting steps and the applicable denominator
        :raises ValueError: If the position is not positive or histogram unavailable
        """
        if position < 1 or self.acceptance_tails is None:
            raise ValueError("A positive position and valid histogram are required")
        tails = self.acceptance_tails
        tail = tails[position - 1] if position <= len(tails) else 0
        previous = tails[position - 2] if 1 < position <= len(tails) + 1 else 0
        denominator = (
            previous if conditional and position > 1 else self.num_spec_steps or 0
        )
        return tail, denominator


class SpeculativeDecodingMetrics(SpeculativeAcceptanceMetrics):
    """Overall acceptance and optional reconstructed step-start phase metrics."""

    reasoning: SpeculativeAcceptanceMetrics | None = Field(
        default=None, description="Steps starting in the reconstructed reasoning prefix"
    )
    content: SpeculativeAcceptanceMetrics | None = Field(
        default=None,
        description="Steps starting after the reconstructed reasoning prefix",
    )

    @classmethod
    def from_response_metrics(
        cls,
        metrics: VLLMRequestMetrics | None,
        *,
        reasoning_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> SpeculativeDecodingMetrics | None:
        """
        Interpret validated vLLM counters without modifying the captured metrics.

        Inconsistent counters are unavailable rather than coerced to zero. Detailed
        phase attribution assumes one initial prefill token and a contiguous reasoning
        prefix. Each subsequent step advances by accepted drafts plus one target token.
        Delimiters and unrecorded non-drafting iterations can affect this approximation.

        :param metrics: Validated server response metrics
        :param reasoning_tokens: Server-reported reasoning prefix length, if available
        :param output_tokens: Server-reported completion length for checking usage
        :return: Interpreted counters, or None when no supported data is available
        """
        raw = metrics.speculative_decoding if metrics is not None else None
        if raw is None:
            return None
        steps = raw.num_spec_steps
        accepted = raw.num_accepted_draft_tokens
        drafted = raw.num_draft_tokens
        if accepted is not None and (
            (drafted is not None and accepted > drafted)
            or (steps == 0 and accepted > 0)
        ):
            accepted = None
        histogram = raw.acceptance_histogram
        if not (
            histogram
            and steps is not None
            and sum(histogram) == steps
            and (
                accepted is None
                or sum(j * count for j, count in enumerate(histogram)) == accepted
            )
        ):
            histogram = None
        reasoning, content = cls._phase_counters(raw, reasoning_tokens, output_tokens)
        if steps is None and accepted is None and drafted is None and reasoning is None:
            return None
        return cls(
            num_spec_steps=steps,
            num_accepted_draft_tokens=accepted,
            num_draft_tokens=drafted,
            acceptance_histogram=histogram,
            reasoning=reasoning,
            content=content,
        )

    @staticmethod
    def _phase_counters(
        raw: VLLMSpeculativeDecodingMetrics,
        reasoning_tokens: int | None,
        output_tokens: int | None,
    ) -> tuple[
        SpeculativeAcceptanceMetrics | None, SpeculativeAcceptanceMetrics | None
    ]:
        accepted = raw.per_step_accepted
        drafted = raw.per_step_drafted
        if (
            accepted is None
            or drafted is None
            or len(accepted) != len(drafted)
            or reasoning_tokens is None
            or output_tokens is None
            or not 0 <= reasoning_tokens <= output_tokens
        ):
            return None, None

        # Count into sparse histograms first, so validation needs only one step scan.
        histograms: list[dict[int, int]] = [{}, {}]
        steps = [0, 0]
        accepted_totals = [0, 0]
        drafted_totals = [0, 0]
        position = 1  # Zero-based offset after the initial non-speculative token.
        max_position = raw.num_spec_tokens or 0
        for num_accepted, num_drafted in zip(accepted, drafted, strict=True):
            if num_accepted > num_drafted or (
                raw.num_spec_tokens is not None and num_drafted > raw.num_spec_tokens
            ):
                return None, None
            phase = 0 if position < reasoning_tokens else 1
            steps[phase] += 1
            accepted_totals[phase] += num_accepted
            drafted_totals[phase] += num_drafted
            histogram = histograms[phase]
            histogram[num_accepted] = histogram.get(num_accepted, 0) + 1
            max_position = max(max_position, num_drafted)
            position += num_accepted + 1

        for reported, observed in (
            (raw.num_spec_steps, len(accepted)),
            (raw.num_accepted_draft_tokens, sum(accepted_totals)),
            (raw.num_draft_tokens, sum(drafted_totals)),
        ):
            if reported is not None and reported != observed:
                return None, None

        if raw.acceptance_histogram is not None:
            histogram_counts = raw.acceptance_histogram
            if sum(histogram_counts) != len(accepted) or any(
                count != histograms[0].get(j, 0) + histograms[1].get(j, 0)
                for j, count in enumerate(histogram_counts)
            ):
                return None, None
            max_position = max(max_position, len(histogram_counts) - 1)

        # Do not reconcile position with output_tokens: server acceptance counters
        # precede terminal truncation. Whole steps retain their original counters.
        phases = [
            SpeculativeAcceptanceMetrics(
                num_spec_steps=steps[phase],
                num_accepted_draft_tokens=accepted_totals[phase],
                num_draft_tokens=drafted_totals[phase],
                acceptance_histogram=[
                    histograms[phase].get(j, 0) for j in range(max_position + 1)
                ],
            )
            for phase in (0, 1)
        ]
        return phases[0], phases[1]
