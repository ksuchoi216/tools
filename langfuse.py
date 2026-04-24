from pathlib import Path
from typing import Any

from loguru import logger

PROMPT_FILE_SUFFIX = ".txt"


def _get_langfuse_client() -> Any:
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise RuntimeError("Langfuse is not installed.") from exc
    return get_client()


def _get_prompt_file_path(prompt_dir: str | Path, prompt_name: str) -> Path:
    return Path(prompt_dir) / f"{prompt_name}{PROMPT_FILE_SUFFIX}"


def load_prompt(
    prompt_name: str, prompt_dir: str | None = None, is_local: bool = False
) -> str:
    if is_local:
        if prompt_dir is None:
            raise ValueError("prompt_dir must be provided when is_local is True.")
        return Path(prompt_dir).read_text(encoding="utf-8")

    prompt = _get_langfuse_client().get_prompt(prompt_name).get_langchain_prompt()
    logger.info("Prompt {} loaded from Langfuse.", prompt_name)
    return prompt


def upload_prompt(prompt_name: str, prompt_text: str):
    langfuse = _get_langfuse_client()
    try:
        langfuse.create_prompt(
            name=prompt_name,
            type="text",
            prompt=prompt_text,
            labels=["production"],
        )
        logger.info("Prompt {} uploaded to Langfuse.", prompt_name)
    except Exception as e:
        logger.error("Failed to upload prompt {}: {}", prompt_name, e)


def download_prompts(prompt_names: list[str], prompt_dir: str) -> dict[str, Path]:
    langfuse = _get_langfuse_client()
    save_dir = Path(prompt_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}
    for prompt_name in prompt_names:
        prompt_text = langfuse.get_prompt(prompt_name).get_langchain_prompt()
        prompt_file = _get_prompt_file_path(save_dir, prompt_name)
        prompt_file.write_text(prompt_text, encoding="utf-8")
        saved_files[prompt_name] = prompt_file
        logger.info("Prompt {} downloaded to {}.", prompt_name, prompt_file)

    return saved_files


def upload_prompts(prompt_names: list[str], prompt_dir: str) -> list[str]:
    langfuse = _get_langfuse_client()

    uploaded_prompt_names: list[str] = []
    for prompt_name in prompt_names:
        prompt_file = _get_prompt_file_path(prompt_dir, prompt_name)
        prompt_text = prompt_file.read_text(encoding="utf-8")
        langfuse.create_prompt(
            name=prompt_name,
            type="text",
            prompt=prompt_text,
            labels=["production"],
        )
        uploaded_prompt_names.append(prompt_name)
        logger.info("Prompt {} uploaded from {}.", prompt_name, prompt_file)

    return uploaded_prompt_names
