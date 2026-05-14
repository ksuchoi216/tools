from enum import Enum


class ModelNames(Enum):
    gpt_41_mini = "gpt-4.1-mini"
    gpt_41 = "gpt-4.1"
    gpt_5_nano = "gpt-5-nano"
    gpt_5_mini = "gpt-5-mini"
    gpt_5 = "gpt-5"
    gpt_54_nano = "gpt-5.4-nano"
    gpt_54_mini = "gpt-5.4-mini"
    gpt_54 = "gpt-5.4"

REASONING_MODELS = {
    ModelNames.gpt_5_nano,
    ModelNames.gpt_5_mini,
    ModelNames.gpt_5,
    ModelNames.gpt_54_nano,
    ModelNames.gpt_54_mini,
    ModelNames.gpt_54,
}