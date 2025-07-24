from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic

from client.api_base import ApiConcreteBase

SYSTEM_PROMPT = (
    "あなたは親切なアシスタントです"
)

USER_PROMPT = (
    "こんにちは\n"
    "{input_1} - {input_2} = ?"
)


class SympleComparisonGrader(ApiConcreteBase):
    def __init__(
            self,
            chat_model,
    ):
        super().__init__(chat_model=chat_model)

    def _invoke_handling(
            self,
            input: Dict[Literal["input_1", "input_2"], str],
            **kwargs
    ):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(input_1=input["input_1"], input_2=input["input_2"])},
        ]

        res = self._chat_model(messages, **kwargs)

        return res.choices[0].message.content


if __name__ == "__main__":
    """
    python -m client.concrete.simple_generator
    """

    import os
    from model.openai_api_llm import OpenAiAPIChatClient

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
        temperature=0.5
    )

    gen = SympleComparisonGrader(llm)

    inst = {"input_1": "10", "input_2": "20"}
    res = gen(input=inst)

    print(res)
