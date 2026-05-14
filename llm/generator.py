from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger
from pandas import read_table
from pydantic import BaseModel

from .chains import build_chain
from .config import OpenAINodeConfig

load_dotenv(find_dotenv(usecwd=True))


class PromptGenerator:
    def __init__(
        self,
        *,
        model_config: OpenAINodeConfig,
        prompt_key: str,
        local_prompt_dir: str | Path | None = None,
        output_parser: type[BaseModel] | None = None,
        node_name: str | None = None,
    ) -> None:
        self.model_config = model_config
        self.prompt_key = prompt_key
        self.local_prompt_dir: str | Path | None = local_prompt_dir
        self.node_name = node_name or self.prompt_key
        self.output_parser = output_parser

        self.initialize()

    def initialize(self):
        self.prompt, self.parser, self.chain = build_chain(
            model_config=self.model_config,
            prompt_key=self.prompt_key,
            local_prompt_dir=self.local_prompt_dir,
            output_parser=self.output_parser,
        )

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

    def _serialize_output(self, output: Any) -> Any:
        if isinstance(self.parser, PydanticOutputParser) and hasattr(
            output, "model_dump"
        ):
            return output.model_dump()
        return output

    def invoke(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> Any:
        prompt_input = self._build_prompt_input(input_data)
        logger.info("Invoking Generator.")
        output = self.chain.invoke(prompt_input, config=config)
        # if output_parser is a PydanticOutputParser, it will use dump
        return self._serialize_output(output)

    def batch(
        self,
        input_data: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> list[Any]:
        batch_inputs = self._build_prompt_inputs(input_data)
        logger.info(
            "Batch invoking Generator with {} prompt inputs.",
            len(batch_inputs),
        )
        outputs = self.chain.batch(batch_inputs, config=config)
        # print outputs

        return [self._serialize_output(output) for output in outputs]


class ChatGenerator(PromptGenerator):
    def _insert_vars_into_template(self, template, variables):
        try:
            return template.format(**variables)
        except KeyError as error:
            missing_key = error.args[0]
            raise ValueError

    def _add_format_instructions_into_template(self, template):
        if isinstance(self.parser, PydanticOutputParser):
            return (
                template + "\n# Output Format\n" + self.parser.get_format_instructions()
            )
        return template

    def build_human_prompt(self) -> str:
        return "generate response according to system prompt."

    def create_image_content_from_image(self, images):
        total = len(images)
        image_content = []
        for index, image in enumerate(images, start=1):
            image_content.append(
                {
                    "type": "text",
                    "text": (
                        f"\n---\n다음은 문서 페이지 {index} 입니다. "
                        f"총 {total} 페이지 중 {index} 페이지입니다.\n---\n"
                    ),
                }
            )
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                        "detail": "auto",
                    },
                }
            )
        return image_content

    def preprocess_input(self, input_data: dict[str, Any]):
        images = input_data.get("images", None)
        if images is not None:
            input_data.pop("images")

        human_prompt = self.build_human_prompt()
        system_content = self._add_format_instructions_into_template(self.prompt)
        system_message = SystemMessage(content=system_content)
        human_prompt = self._insert_vars_into_template(human_prompt, input_data)
        human_content = [{"type": "text", "text": human_prompt}]

        if images is not None:
            # TODO: check base64 format.
            image_contents = self.create_image_content_from_image(images)
            human_content.extend(image_contents)

        human_message = HumanMessage(content=human_content)
        return [system_message, human_message]
