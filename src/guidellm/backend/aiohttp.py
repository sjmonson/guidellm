from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from loguru import logger

import aiohttp
import json

from guidellm.backend.base import Backend, GenerativeResponse
from guidellm.config import settings
from guidellm.core import TextGenerationRequest

__all__ = ["AiohttpBackend"]

def deepget(obj: Union[dict, list], *path: Any, default: Any = None) -> Any:
    """
    Acts like .get() but for nested objects.

    Each item in path is recusively indexed on obj. For path of length N,
      obj[path[0]][path[1]]...[path[N-1]][path[N]]

    :param obj: root object to index
    :param path: ordered list of keys to index recursively
    :param default: the default value to return if an indexing fails
    :returns: result of final index or default if Key/Index Error occurs
    """
    current = obj
    for pos in path:
        try:
            current = current[pos]
        except (KeyError, IndexError):
            return default
    return current

@Backend.register("aiohttp_server")
class AiohttpBackend(Backend):
    """
    An aiohttp-based backend implementation for LLM requests.

    This class provides an interface to communicate with a server hosting
    an LLM API using aiohttp for asynchronous requests.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        target: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        **request_args,
    ):
        self._request_args: Dict = request_args        
        self._api_key: str = openai_api_key or settings.aiohttp.api_key

        if not self._api_key:
            err = ValueError(
                "`GUIDELLM__AIOHTTP__API_KEY` environment variable or "
                "--openai-api-key CLI parameter must be specified for the "
                "aiohttp backend."
            )
            logger.error("{}", err)
            raise err

        base_url = target or settings.aiohttp.base_url
        self._api_url = f"{base_url}/chat/completions"

        if not base_url:
            err = ValueError(
                "`GUIDELLM__AIOHTTP__BASE_URL` environment variable or "
                "target parameter must be specified for the OpenAI backend."
            )
            logger.error("{}", err)
            raise err

        self._timeout = aiohttp.ClientTimeout(total=timeout or settings.request_timeout)
        self._model = model

        super().__init__(type_="aiohttp_backend", target=base_url, model=self._model)
        logger.info("aiohttp {} Backend listening on {}", self._model, base_url)

    async def make_request(
        self,
        request: TextGenerationRequest,
    ) -> AsyncGenerator[GenerativeResponse, None]:
        """
        Make a request to the aiohttp backend.

        Sends a prompt to the LLM server and streams the response tokens.

        :param request: The text generation request to submit.
        :type request: TextGenerationRequest
        :yield: A stream of GenerativeResponse objects.
        :rtype: AsyncGenerator[GenerativeResponse, None]
        """

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            logger.debug("Making request to aiohttp backend with prompt: {}", request.prompt)

            request_args = {}
            if request.output_token_count is not None:
                request_args.update(
                    {
                        "max_tokens": request.output_token_count,
                        "stop": None,
                        "ignore_eos": True,
                    }
                )
            elif settings.aiohttp.max_gen_tokens and settings.aiohttp.max_gen_tokens > 0:
                request_args.update(
                    {
                        "max_tokens": settings.aiohttp.max_gen_tokens,
                    }
                )

            request_args.update(self._request_args)

            payload = {
                "model": self._model,
                "messages": [
                    {"role": "user", "content": request.prompt},
                ],
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                    "continuous_usage_stats": True,
                },
                **request_args,
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            }

            try:
                async with session.post(url=self._api_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_message = await response.text()
                        logger.error("Request failed: {} - {}", response.status, error_message)
                        raise Exception(f"Failed to generate response: {error_message}")

                    total_usage = 0
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk == "[DONE]":
                            # Final response
                            yield GenerativeResponse(
                                type_="final",
                                prompt=request.prompt,
                                output_token_count=total_usage,
                                prompt_token_count=request.prompt_token_count,
                            )
                            continue

                        message = json.loads(chunk)
                        token = {}

                        token["text"] = deepget(message, "choices", 0, 'delta', 'content')

                        # If the message has the current usage then record the number of
                        # tokens, otherwise assume 1 token
                        current_usage = deepget(message, "usage", "completion_tokens")
                        if current_usage != None:
                            token['count'] = current_usage - total_usage
                        else:
                            token['count'] = 1

                        # Omit responses that don't have
                        # tokens (or somehow negative tokens)
                        if token['count'] < 1:
                            logger.debug("Omiting response '%s' because it contains %d tokens",
                                        token["text"], token['count'])
                            continue

                        for _ in range(token['count']):
                            # Update the total token count
                            total_usage += 1
                            yield GenerativeResponse(
                                type_="token_iter",
                                add_token=token["text"],
                                prompt=request.prompt,
                                output_token_count=total_usage,
                                prompt_token_count=request.prompt_token_count,
                            )

            except Exception as e:
                logger.error("Error while making request: {}", e)
                raise

    def available_models(self) -> List[str]:
        """
        Retrieve a list of available models from the server.
        """
        # This could include an API call to `self._api_url/models` if the server supports it.
        logger.warning("Fetching available models is not implemented for aiohttp backend.")
        return []

    def validate_connection(self):
        """
        Validate the connection to the backend server.
        """
        logger.info("Connection validation is not explicitly implemented for aiohttp backend.")
