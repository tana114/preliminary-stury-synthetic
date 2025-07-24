import os

from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

# これはOpenAIではなくOpenRouterのモデルのリストです
valid_model_names = Literal[
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen3-235b-a22b-07-25:free",
]

from openai import OpenAI, OpenAIError

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())


class OpenAiAPIChatClient:
    def __init__(
            self,
            model_name: Literal[valid_model_names],
            base_url: str,
            api_key: str,
            **base_params,
    ):
        self._model = model_name
        self._base_params = base_params
        self._base_url = base_url
        self._api_key = api_key
        # <class 'openai.OpenAI'>
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(
            self,
            messages: List[Dict[str, str]],
            **override_params
    ):
        params = {
            "model": self._model,
            **self._base_params,
            "messages": messages,
            **override_params  # 実行時に渡されたパラメータが優先
        }

        # Noneの値を削除
        params = {k: v for k, v in params.items() if v is not None}

        # <class 'openai.types.chat.chat_completion.ChatCompletion'>
        return self._client.chat.completions.create(**params)


if __name__ == "__main__":
    """
    python -m model.api_llm_base
    """
    from dotenv import load_dotenv

    load_dotenv()

    # api_key = "your_api_key_here"
    # api_key = os.environ.get("OPENAI_API_KEY")
    # base_url = "https://api.openai.com/v1"

    # api_key = "your_api_key_here"
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"

    llm = OpenAiAPIChatClient(
        model_name="deepseek/deepseek-r1:free",
        api_key=api_key,
        base_url=base_url,
        max_tokens=500,
        temperature=0.5
    )

    messages = [{"role": "user", "content": "こんにちは！"}]

    res = llm(messages=messages, max_tokens=1000)

    print(type(res))
    print(res)
