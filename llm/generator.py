from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from loguru import logger
from pydantic import BaseModel

from .chains import build_chain
from .config import OpenAINodeConfig

load_dotenv(find_dotenv(usecwd=True))


class Generator:
    def __init__(
        self,
        *,
        model_config: OpenAINodeConfig,
        prompt_key: str,
        local_prompt: bool = False,
        local_prompt_dir: str | Path | None = None,
        output_parser: type[BaseModel] | None = None,
        node_name: str | None = None,
    ) -> None:
        self.output_parser = output_parser
        self.model_config = model_config
        self.prompt_key = prompt_key
        self.local_prompt = local_prompt
        self.local_prompt_dir: str | Path | None = local_prompt_dir
        self.node_name = node_name or self.prompt_key
        self.prompt, self.parser, self.chain = build_chain(
            model_config=model_config,
            prompt_key=prompt_key,
            local_prompt=local_prompt,
            local_prompt_dir=local_prompt_dir,
            output_parser=output_parser,
        )
        logger.info(
            "Initialized PromptGenerator: model={}, prompt_key={}, prompt_source={}, has_output_parser={}",
            self.model_config.model_name,
            self.prompt_key,
            "local" if self.local_prompt else "langfuse",
            output_parser is not None,
        )
        self.set_input_keys()

    def set_input_keys(self):
        self.input_keys = None

    def verify_input(self, input_data: dict[str, Any]) -> bool:
        if not hasattr(self, "input_keys"):
            raise ValueError("input_keys is not set in Generator")
        if self.input_keys is None:
            return True
        missing_keys = [key for key in self.input_keys if key not in input_data]
        if missing_keys:
            logger.warning(
                "Input data is missing keys required by Generator {}: {}",
                self.node_name,
                missing_keys,
            )
            return False
        return True

    # OVERRIDE POSSIBLELY
    def preprocess_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        return input_data

    # OVERRIDE POSSIBLELY
    def preprocess_inputs(self, input_data: dict[str, Any]) -> list[dict[str, Any]]:
        return [input_data]

    def _build_prompt_input(self, input_data: dict[str, Any]) -> dict[str, Any]:
        prompt_input = self.preprocess_input(input_data)
        if not isinstance(prompt_input, dict):
            raise TypeError("preprocess_input() must return a dict.")

        if isinstance(self.parser, PydanticOutputParser):
            prompt_input = {
                **prompt_input,
                "format_instructions": self.parser.get_format_instructions(),
            }
        return prompt_input

    def _build_prompt_inputs(self, input_data: dict[str, Any]) -> list[dict[str, Any]]:
        batch_input_data = self.preprocess_inputs(input_data)
        if not isinstance(batch_input_data, list) or not all(
            isinstance(item, dict) for item in batch_input_data
        ):
            raise TypeError("preprocess_inputs() must return a list of dicts.")

        return [self._build_prompt_input(item) for item in batch_input_data]

    def invoke(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> Any:
        if not self.verify_input(input_data):
            raise ValueError(
                f"Input data is missing required keys for {self.node_name}."
            )

        prompt_input = self._build_prompt_input(input_data)
        logger.info("Invoking Generator.")

        output = self.chain.invoke(prompt_input, config=config)
        # if output_parser is a PydanticOutputParser, it will use dump
        if isinstance(self.parser, PydanticOutputParser) and hasattr(
            output, "model_dump"
        ):
            output = output.model_dump()

        return output

    def batch(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> list[Any]:
        if not self.verify_input(input_data):
            raise ValueError(
                f"Input data is missing required keys for {self.node_name}."
            )

        batch_inputs = self._build_prompt_inputs(input_data)
        logger.info(
            "Batch invoking Generator with {} prompt inputs.",
            len(batch_inputs),
        )
        outputs = self.chain.batch(batch_inputs, config=config)
        output = outputs[0]
        if isinstance(self.parser, PydanticOutputParser) and hasattr(
            output, "model_dump"
        ):
            outputs = [output.model_dump() for output in outputs]

        return outputs
