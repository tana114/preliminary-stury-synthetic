from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any, TypeVar, Generic

# from client.api_base import ApiConcreteBase
from client.vllm_client_base import VllmConcreteBase

from vllm import LLM, SamplingParams

USER_PROMPT = (
    "こんにちは\n"
    "{input_1} - {input_2} = ?"
)


class SympleComparisonGrader(VllmConcreteBase):
    def __init__(
            self,
            chat_model,
    ):
        super().__init__(chat_model=chat_model)

    def _invoke_handling(
            self,
            input: Dict[Literal["input_1", "input_2"], str],
            sampling_params: SamplingParams,
    ) -> List[str]:
        messages = USER_PROMPT.format(input_1=input["input_1"], input_2=input["input_2"])
        return self._chat_model.generate(messages, sampling_params)


if __name__ == "__main__":
    """
    python -m client.concrete.vllm_simple_gen
    """

    import os
    import torch.distributed as dist
    from vllm import LLM, SamplingParams
    from model.vllm_llm_base import VllmClient

    # 推論モデルのロード
    print("推論モデルをロード中...")
    llm = VllmClient(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=0.95,
        dtype="half",  # float16 を使用古いGPU用
    )

    prompts = "hello!!"

    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=1024,
        n=1,  # 各プロンプトに対する生成数
    )

    gen = SympleComparisonGrader(llm)

    inst = {"input_1": "10", "input_2": "20"}
    res = gen(input=inst, sampling_params=sampling_params)

    print(type(res))
    print(len(res))

    if dist.is_initialized():
        dist.destroy_process_group()  # プロセスグループのクリーンアップ
