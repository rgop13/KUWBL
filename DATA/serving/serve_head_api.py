# ray head 에서 실행해야 함
import os
import argparse
from pathlib import Path
from typing import List, Any, Tuple

from tqdm.asyncio import tqdm as tqdm_asyncio

from ray import serve
from ray.serve.llm import (
    LLMConfig,
    build_openai_app,
    ModelLoadingConfig,
)

import re
# ① 정상 쌍 (<think> … </think>)
THINK_PAIR = re.compile(
    r"<\s*think\b[^>]*>.*?</\s*think\s*>\s*\n{0,2}",  # \n 또는 \n\n 허용
    flags=re.S | re.I,
)

# ② 닫는 태그가 없고 대신 '\n\n' 로 끝나는 경우
THINK_OPEN_TILL_BLANK = re.compile(
    r"<\s*think\b[^>]*>.*?\n{2}",
    flags=re.S | re.I,
)
def strip_think(text: str) -> str:
    text = re.sub(THINK_PAIR, "", text)
    text = re.sub(THINK_OPEN_TILL_BLANK, "", text)
    return text.lstrip()


class VLLMRayHeadServer:
    def __init__(self,
                 model_name: str,
                 max_model_len: int,
                 tensor_parallel_size: int = 4,
                 pipeline_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.95,
                 **kwargs):
        # Local snapshot 강제
        revision = self.get_deepseek_snapshot_revision()
        vllm_engine_args = {
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "download_dir": "/data/data_team/cache/huggingface",
            "revision": revision,
            "tokenizer_revision": revision,
            **kwargs,
        }
        # if "qwen3" in model_name.lower():
            # vllm_engine_args["reasoning_parser"] = "qwen"
            # vllm_engine_args["enable_reasoning"] = True
        v3_config = LLMConfig(
            model_loading_config=ModelLoadingConfig(
                model_id=model_name,
                model_source=model_name,
                tokenizer_source=model_name,
            ),
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "max_replicas": 1,  # Data Parallelism임 -> 처리 속도 관점
                }
            },
            runtime_env={
                "pip": [
                    "pyarrow==21.0.0",
                    "pandas==2.3.2",
                    "lz4",
                    "jsonlines",
                    "fsspec"
                ],
                "env_vars": {
                    # HF 캐시
                    "HF_HOME": "/data/data_team/cache/huggingface",
                    "HF_HUB_CACHE": "/data/data_team/cache/huggingface",
                    "TRANSFORMERS_CACHE": "/data/data_team/cache/huggingface/transformers",
                    "HF_DATASETS_CACHE": "/data/data_team/cache/huggingface/datasets",
                    "VLLM_DOWNLOAD_DIR": "/data/data_team/cache/huggingface",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    # 통신 관련
                    "NCCL_SOCKET_IFNAME": "eth0",
                    "GLOO_SOCKET_IFNAME": "eth0",
                    "NCCL_SOCKET_FAMILY": "AF_INET",
                    "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT,NET",
                    "TORCH_NCCL_BLOCKING_WAIT": "1",
                    "NCCL_ASYNC_ERROR_HANDLING": "1",
                    "GLOO_USE_LIBUV": "0",
                    "GLOO_USE_IPV6": "0",
                    "TP_USE_IPV6": "0",
                    "VLLM_USE_V1": "1",
                    "TF_ENABLE_ONEDNN_OPTS": "0",
                    "NUMEXPR_MAX_THREADS": "64",
                    "RAY_TMPDIR": "/data/ray_tmp",
                    "RAY_VERBOSE": "1",
                }
            },
            accelerator_type="H100",
            engine_kwargs=vllm_engine_args,
        )
        
        llm_app = build_openai_app(
            {"llm_configs": [v3_config]}
        )
        serve.run(llm_app)
        
    def get_deepseek_snapshot_revision(self) -> str:
        HF_HOME = os.environ.get("HF_HOME", "/data/data_team/cache/huggingface")
        repo_cache = Path(HF_HOME) / "models--deepseek-ai--DeepSeek-V3.1"
        with open(repo_cache / "refs" / "main", "r") as f:
            REVISION = f.read().strip()
        return REVISION

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-V3.1")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--pipeline_parallel_size", type=int, default=2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_num_seqs", type=int, default=64)
    args = parser.parse_args()
    generator = VLLMRayHeadServer(
        model_name=args.model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        trust_remote_code=True,
        enable_expert_parallel=True,
        enable_prefix_caching=True,
    )