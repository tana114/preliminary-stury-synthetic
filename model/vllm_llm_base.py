from typing import Generator, Dict, Optional, Literal, TypedDict, List, Any

# これは参照用のvllmモデルのリストです。実際にはこれ以外のモデルも使用できます。
valid_model_names = Literal[
    "Qwen/Qwen2.5-0.5B-Instruct",
]

from logging import getLogger, NullHandler

logger = getLogger(__name__)
logger.addHandler(NullHandler())

import os

# vllmにbasicConfigは効かない。環境変数で指定する必要がある
# os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
# os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

from vllm import LLM, SamplingParams


class VllmClient(LLM):
    def __init__(
            self,
            model_name: Literal[valid_model_names],
            tensor_parallel_size: int = 1,
            pipeline_parallel_size: int = 1,
            data_parallel_size: int = 1,
            gpu_memory_utilization: float = 0.95,
            **base_params,
    ):
        # self._model = model_name
        # self._tensor_parallel_size = tensor_parallel_size
        # self._pipeline_parallel_size = pipeline_parallel_size
        # self._data_parallel_size = data_parallel_size
        # self._gpu_memory_utilization = gpu_memory_utilization
        # self._base_params = base_params

        shared_kwargs: dict = dict(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **base_params,
        )

        super().__init__(**shared_kwargs)

    def __call__(
            self,
            messages: str,
            sampling_params: SamplingParams
    ):
        return self.generate(messages, sampling_params)


if __name__ == "__main__":
    """
    python -m model.vllm_llm_base
    """
    from vllm import LLM, SamplingParams
    import torch.distributed as dist

    llm = VllmClient(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="half",  # float16 を使用古いGPU用
    )

    prompts = "hello!!"

    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=1024,
        n=1,  # 各プロンプトに対する生成数
    )

    try:
        outputs = llm(prompts, sampling_params)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()  # プロセスグループのクリーンアップ

    print(type(outputs))
    print(len(outputs))
