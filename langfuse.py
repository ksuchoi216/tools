from typing import Optional

from langfuse import get_client
from loguru import logger


def check_langfuse_available():
    langfuse = get_client()
    if langfuse.auth_check():
        logger.info("Langfuse is available and authenticated.")
        return True, langfuse
    else:
        logger.warning("Langfuse is not available or authentication failed.")
        return False, None


def load_prompt(
    prompt_name: str, prompt_dir: Optional[str] = None, is_local: bool = False
):
    if is_local:
        if prompt_dir is None:
            raise ValueError("prompt_dir must be provided when is_local is True.")
        prompt = open(prompt_dir, "r").read()
        return prompt
    else:
        prompt = get_client().get_prompt(prompt_name).get_langchain_prompt()
        logger.info("Prompt {} loaded from Langfuse.", prompt_name)
        return prompt


def upload_prompt(prompt_name: str, prompt_dir: str):
    # upload prompt from local file system(prompt_dir) to langfuse with prompt_name
    is_langfuse_available, langfuse = check_langfuse_available()
    if is_langfuse_available:
        prompt_text = open(prompt_dir, "r").read()
        langfuse.create_prompt(
            name=prompt_name,
            type="text",
            prompt=prompt_text,
            labels=["production"],  # optionally, directly promote to production
        )
        logger.info("Prompt {} uploaded to Langfuse from {}.", prompt_name, prompt_dir)
    else:
        logger.error(
            "Failed to upload prompt {} to Langfuse. Langfuse is not available.",
            prompt_name,
        )
        raise RuntimeError("Langfuse is not available. Cannot upload prompt.")
