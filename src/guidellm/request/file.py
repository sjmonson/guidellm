import json
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from transformers import PreTrainedTokenizer  # type: ignore  # noqa: PGH003

from guidellm.config import settings
from guidellm.core.request import TextGenerationRequest
from guidellm.request.base import GenerationMode, RequestGenerator
from guidellm.utils import load_text_lines

__all__ = ["FileRequestGenerator"]


class FileRequestGenerator(RequestGenerator):
    """
    A request generator implementation for files.

    :param path: The path to the file containing the data.
    :type path: Optional[Union[str, Path]]
    :param tokenizer: The tokenizer instance or the name/config to use
        for tokenizing prompts.
    :type tokenizer: Union[str, PreTrainedTokenizer]
    :param mode: The generation mode, either 'async' or 'sync'.
    :type mode: str
    :param async_queue_size: The size of the request queue.
    :type async_queue_size: int
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]],
        tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
        mode: GenerationMode = "async",
        async_queue_size: int = 50,
    ):
        if not path:
            raise ValueError("File path must be provided for FileRequestGenerator")
        self._path = path
        self._data = []
        with open(path, "r", encoding="utf-8") as file:
            # [1:] to skip the first line, it contains metadata
            lines = file.readlines()[1:]
            for line in lines:
                # Load each line as a JSON object
                try:
                    json_object = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    logger.error("Error decoding JSON in file %s %s", path, e)
                    continue
                try:
                    input_tokens = int(json_object["tok_input_length"])
                    output_tokens = int(json_object["tok_output_length"])
                    prompt = json_object["question"]
                    input_id = json_object["index"]
                except KeyError as e:
                    logger.error(
                        "Unexpected format in dataset file %s, KeyError: %s, \n %s", path, e, json_object
                    )
                    continue
                    # TODO exit or just skip here?

                input_data = {
                    "text": prompt,
                    "input_id": input_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
                self._data.append(input_data)

        self._iterator = iter(self._data)

        # NOTE: Must be after all the parameters since the queue population
        #       function requires attributes above
        super().__init__(
            type_="file",
            source=str(path),
            tokenizer=tokenizer,
            mode=mode,
            async_queue_size=async_queue_size,
        )

    def __len__(self) -> int:
        """
        Return the number of text lines.
        """

        return len(self._data)

    def create_item(self) -> TextGenerationRequest:
        """
        Create a new result request item from the data.

        :return: A new result request.
        :rtype: TextGenerationRequest
        """
        logger.debug("Creating new request item from file data")

        try:
            data = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._data)
            data = next(self._iterator)

        request = TextGenerationRequest(
            prompt=data["text"],
            prompt_token_count=data["input_tokens"],
            output_token_count=data["output_tokens"],
        )
        logger.debug("Created new TextGenerationRequest: {}", request)

        return request
