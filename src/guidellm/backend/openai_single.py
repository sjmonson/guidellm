import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal, Optional, Union

from loguru import logger

from guidellm.backend.backend import Backend
from guidellm.backend.openai import (
    CHAT_COMPLETIONS_PATH,
    TEXT_COMPLETIONS_PATH,
    OpenAIHTTPBackend,
)
from guidellm.backend.response import (
    RequestArgs,
    ResponseSummary,
    StreamingTextResponse,
)

__all__ = [
    "OpenAISingleHTTPBackend",
]

EndpointType = Literal["chat_completions", "models", "text_completions"]


@Backend.register("openai_single_http")
class OpenAISingleHTTPBackend(OpenAIHTTPBackend):
    """
    A HTTP-based backend implementation for requests to an OpenAI compatible server.
    For example, a vLLM server instance or requests to OpenAI's API.

    :param target: The target URL string for the OpenAI server. ex: http://0.0.0.0:8000
    :param model: The model to use for all requests on the target server.
        If none is provided, the first available model will be used.
    :param api_key: The API key to use for requests to the OpenAI server.
        If provided, adds an Authorization header with the value
        "Authorization: Bearer {api_key}".
        If not provided, no Authorization header is added.
    :param organization: The organization to use for requests to the OpenAI server.
        For example, if set to "org_123", adds an OpenAI-Organization header with the
        value "OpenAI-Organization: org_123".
        If not provided, no OpenAI-Organization header is added.
    :param project: The project to use for requests to the OpenAI server.
        For example, if set to "project_123", adds an OpenAI-Project header with the
        value "OpenAI-Project: project_123".
        If not provided, no OpenAI-Project header is added.
    :param timeout: The timeout to use for requests to the OpenAI server.
        If not provided, the default timeout provided from settings is used.
    :param http2: If True, uses HTTP/2 for requests to the OpenAI server.
        Defaults to True.
    :param follow_redirects: If True, the HTTP client will follow redirect responses.
        If not provided, the default value from settings is used.
    :param max_output_tokens: The maximum number of tokens to request for completions.
        If not provided, the default maximum tokens provided from settings is used.
    :param extra_query: Query parameters to include in requests to the OpenAI server.
        If "chat_completions", "models", or "text_completions" are included as keys,
        the values of these keys will be used as the parameters for the respective
        endpoint.
        If not provided, no extra query parameters are added.
    """

    def __init__(
        self,
        target: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        timeout: Optional[float] = None,
        http2: Optional[bool] = True,
        follow_redirects: Optional[bool] = None,
        max_output_tokens: Optional[int] = None,
        extra_query: Optional[dict] = None,
        extra_body: Optional[dict] = None,
    ):
        super().__init__(
            target=target,
            model=model,
            api_key=api_key,
            organization=organization,
            project=project,
            timeout=timeout,
            http2=http2,
            follow_redirects=follow_redirects,
            max_output_tokens=max_output_tokens,
            extra_query=extra_query,
            extra_body=extra_body,
            type_="openai_single_http",
        )

    def _completions_payload(
        self,
        body: Optional[dict],
        orig_kwargs: Optional[dict],
        max_output_tokens: Optional[int],
        **kwargs,
    ) -> dict:
        payload = body or {}
        payload.update(orig_kwargs or {})
        payload.update(kwargs)
        payload["model"] = self.model

        if max_output_tokens or self.max_output_tokens:
            logger.debug(
                "{} adding payload args for setting output_token_count: {}",
                self.__class__.__name__,
                max_output_tokens or self.max_output_tokens,
            )
            payload["max_tokens"] = max_output_tokens or self.max_output_tokens
            payload["max_completion_tokens"] = payload["max_tokens"]

            if max_output_tokens:
                # only set stop and ignore_eos if max_output_tokens set at request level
                # otherwise the instance value is just the max to enforce we stay below
                payload["stop"] = None
                payload["ignore_eos"] = True

        return payload

    async def _iterative_completions_request(
        self,
        type_: Literal["text_completions", "chat_completions"],
        request_id: Optional[str],
        request_prompt_tokens: Optional[int],
        request_output_tokens: Optional[int],
        headers: dict[str, str],
        params: dict[str, str],
        payload: dict[str, Any],
    ) -> AsyncGenerator[Union[StreamingTextResponse, ResponseSummary], None]:
        if type_ == "text_completions":
            target = f"{self.target}{TEXT_COMPLETIONS_PATH}"
        elif type_ == "chat_completions":
            target = f"{self.target}{CHAT_COMPLETIONS_PATH}"
        else:
            raise ValueError(f"Unsupported type: {type_}")

        logger.info(
            "{} making request: {} to target: {} using http2: {} following "
            "redirects: {} for timeout: {} with headers: {} and params: {} and ",
            "payload: {}",
            self.__class__.__name__,
            request_id,
            target,
            self.http2,
            self.follow_redirects,
            self.timeout,
            headers,
            params,
            payload,
        )

        response_value = ""
        response_prompt_count: Optional[int] = None
        response_output_count: Optional[int] = None
        iter_count = 0
        start_time = time.time()
        iter_time = start_time
        first_iter_time: Optional[float] = None
        last_iter_time: Optional[float] = None

        yield StreamingTextResponse(
            type_="start",
            value="",
            start_time=start_time,
            first_iter_time=None,
            iter_count=iter_count,
            delta="",
            time=start_time,
            request_id=request_id,
        )

        # reset start time after yielding start response to ensure accurate timing
        start_time = time.time()

        resp = await self._get_async_client().post(
            target, headers=headers, params=params, json=payload
        )
        resp.raise_for_status()

        iter_time = time.time()
        line = resp.text
        logger.debug(
            "{} request: {} recieved iter response line: {}",
            self.__class__.__name__,
            request_id,
            line,
        )

        data = json.loads(line.strip())
        if delta := self._extract_completions_delta_content(type_, data):
            pass
        else:
            delta = ""
        first_iter_time = iter_time
        last_iter_time = iter_time
        iter_count += 1
        response_value += delta

        yield StreamingTextResponse(
            type_="iter",
            value=response_value,
            iter_count=iter_count,
            start_time=start_time,
            first_iter_time=first_iter_time,
            delta=delta,
            time=iter_time,
            request_id=request_id,
        )

        if usage := self._extract_completions_usage(data):
            response_prompt_count = usage["prompt"]
            response_output_count = usage["output"]

        logger.info(
            "{} request: {} with headers: {} and params: {} and payload: {} completed"
            "with: {}",
            self.__class__.__name__,
            request_id,
            headers,
            params,
            payload,
            response_value,
        )

        yield ResponseSummary(
            value=response_value,
            request_args=RequestArgs(
                target=target,
                headers=headers,
                params=params,
                payload=payload,
                timeout=self.timeout,
                http2=self.http2,
                follow_redirects=self.follow_redirects,
            ),
            start_time=start_time,
            end_time=iter_time,
            first_iter_time=first_iter_time,
            last_iter_time=last_iter_time,
            iterations=iter_count,
            request_prompt_tokens=request_prompt_tokens,
            request_output_tokens=request_output_tokens,
            response_prompt_tokens=response_prompt_count,
            response_output_tokens=response_output_count,
            request_id=request_id,
        )
