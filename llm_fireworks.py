import llm
from llm.default_plugins.openai_models import Chat

MODELS = (
    "accounts/fireworks/models/firellava-13b",
    "accounts/fireworks/models/firefunction-v1",
    "accounts/fireworks/models/mixtral-8x7b-instruct",
    "accounts/fireworks/models/mixtral-8x22b-instruct",
    "accounts/fireworks/models/llama-v3-70b-instruct",
    "accounts/fireworks/models/bleat-adapter",
    "accounts/fireworks/models/chinese-llama-2-lora-7b",
    "accounts/fireworks/models/dbrx-instruct",
    "accounts/fireworks/models/gemma-7b-it",
    "accounts/fireworks/models/hermes-2-pro-mistral-7b",
    "accounts/fireworks/models/llama-2-13b-fp16-french",
    "accounts/fireworks/models/llama-2-13b-guanaco-peft",
    "accounts/fireworks/models/llama2-7b-summarize",
    "accounts/fireworks/models/llama-guard-2-8b",
    "accounts/fireworks/models/llama-v2-13b",
    "accounts/fireworks/models/llama-v2-13b-chat",
    "accounts/fireworks/models/llama-v2-13b-code",
    "accounts/fireworks/models/llama-v2-13b-code-instruct",
    "accounts/fireworks/models/llama-v2-34b-code",
    "accounts/fireworks/models/llama-v2-34b-code-instruct",
    "accounts/fireworks/models/llama-v2-70b-chat",
    "accounts/fireworks/models/llama-v2-70b-code-instruct",
    "accounts/fireworks/models/llama-v2-7b",
    "accounts/fireworks/models/llama-v2-7b-chat",
    "accounts/fireworks/models/llama-v3-8b-instruct",
    "accounts/fireworks/models/llava-yi-34b",
    "accounts/fireworks/models/mistral-7b",
    "accounts/fireworks/models/mistral-7b-instruct-4k",
    "accounts/fireworks/models/mistral-7b-instruct-v0p2",
    "accounts/fireworks/models/mixtral-8x22b-hf",
    "accounts/fireworks/models/mixtral-8x22b-instruct-hf",
    "accounts/fireworks/models/mixtral-8x7b",
    "accounts/fireworks/models/mixtral-8x7b-instruct-hf",
    "accounts/fireworks/models/mythomax-l2-13b",
    "accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo-fp8",
    "accounts/fireworks/models/openorca-7b",
    "accounts/fireworks/models/qwen-14b-chat",
    "accounts/fireworks/models/qwen-72b-chat",
    "accounts/fireworks/models/starcoder-16b",
    "accounts/fireworks/models/starcoder-7b",
    "accounts/fireworks/models/traditional-chinese-qlora-llama2",
    "accounts/fireworks/models/yi-34b-200k-capybara",
    "accounts/fireworks/models/zephyr-7b-beta",
)


class FireworksChat(Chat):
    needs_key = "fireworks"

    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            model_id=(
                model_name[len("accounts/") :]
                if model_name.startswith("accounts/")
                else model_name
            ),
            api_base="https://api.fireworks.ai/inference/v1",
        )

    def __str__(self):
        return "Fireworks: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only do this if the key is set
    key = llm.get_key("", "fireworks", "LLM_FIREWORKS_KEY")
    if not key:
        return

    for model_id in MODELS:
        register(FireworksChat(model_id))
