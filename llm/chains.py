from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from .config import OpenAINodeConfig
from .langfuse import load_prompt
from .models import ModelNames

ParserType = PydanticOutputParser | StrOutputParser
REASONING_MODELS = {
    ModelNames.gpt_5_nano,
    ModelNames.gpt_5_mini,
    ModelNames.gpt_5,
    ModelNames.gpt_54_nano,
    ModelNames.gpt_54_mini,
    ModelNames.gpt_54,
}


def build_prompt(
    *,
    prompt_text: str | None = None,
    prompt_key: str | None = None,
    local_prompt: bool = False,
    local_prompt_dir: str | Path | None = None,
) -> BasePromptTemplate:
    if prompt_text is not None and prompt_key is not None:
        raise ValueError("Provide either prompt_text or prompt_key, not both.")
    if prompt_text is None and prompt_key is None:
        raise ValueError("Either prompt_text or prompt_key must be provided.")
    if prompt_text is not None:
        return PromptTemplate.from_template(prompt_text)
    assert prompt_key is not None
    return load_prompt(
        prompt_key,
        prompt_dir=local_prompt_dir,
        is_local=local_prompt,
    )


def validate_model_name(model_name: str) -> ModelNames:
    try:
        return ModelNames(model_name)
    except ValueError as exc:
        supported_models = ", ".join(model.value for model in ModelNames)
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Supported models: {supported_models}"
        ) from exc


def build_llm(model_config: OpenAINodeConfig) -> ChatOpenAI:
    model_name = validate_model_name(model_config.model_name)

    llm_kwargs: dict[str, Any] = {
        "model": model_config.model_name,
        "use_responses_api": model_config.use_responses_api,
    }
    if model_name in REASONING_MODELS:
        if model_config.temperature is not None:
            raise ValueError("temperature is only supported for non-reasoning models.")
        if model_config.reasoning is not None:
            llm_kwargs["reasoning"] = model_config.reasoning.model_dump()
        if model_config.verbosity is not None:
            if model_config.use_responses_api:
                llm_kwargs["model_kwargs"] = {
                    "text": {"verbosity": model_config.verbosity}
                }
            else:
                llm_kwargs["verbosity"] = model_config.verbosity
    else:
        if model_config.reasoning is not None or model_config.verbosity is not None:
            raise ValueError(
                "reasoning and verbosity are only supported for reasoning models."
            )
        if model_config.temperature is not None:
            llm_kwargs["temperature"] = model_config.temperature

    if model_config.prompt_cache_key:
        llm_kwargs["extra_body"] = {"prompt_cache_key": model_config.prompt_cache_key}
    return ChatOpenAI(**llm_kwargs)


def build_bound_llm(
    model_config: OpenAINodeConfig,
    *,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
):
    llm = build_llm(model_config)
    if not tools:
        return llm

    bind_kwargs: dict[str, Any] = {}
    if tool_choice is not None:
        bind_kwargs["tool_choice"] = tool_choice
    return llm.bind_tools(list(tools), **bind_kwargs)


def build_output_parser(output_parser) -> ParserType:
    if isinstance(output_parser, type) and issubclass(output_parser, BaseModel):
        logger.info(
            "Using PydanticOutputParser with model: {}",
            output_parser.__name__,
        )
        return PydanticOutputParser(pydantic_object=output_parser)
    logger.info("Using StrOutputParser for output parsing.")
    return StrOutputParser()


def build_chain(
    *,
    model_config: OpenAINodeConfig,
    prompt_key: str,
    local_prompt: bool = False,
    local_prompt_dir: str | Path | None = None,
    output_parser=None,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
) -> tuple[BasePromptTemplate, ParserType, Any]:
    prompt = build_prompt(
        prompt_key=prompt_key,
        local_prompt=local_prompt,
        local_prompt_dir=local_prompt_dir,
    )
    llm = build_bound_llm(
        model_config,
        tools=tools,
        tool_choice=tool_choice,
    )
    parser = build_output_parser(output_parser)
    return prompt, parser, prompt | llm | parser
