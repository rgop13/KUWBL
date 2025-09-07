import os
import sys
import asyncio
import logging
import jsonlines
from typing import List, Tuple, Any, Dict, Optional

from openai import AsyncOpenAI
from httpx import Limits, Timeout, AsyncClient
from tqdm import tqdm  # 진행률 바

class AsyncGenerator:
    def __init__(self, model_name: str, host: str, *, timeout_s: float = 60.0, max_retries: int = 2):
        base_url = self._canon_base_url(host)

        # HTTP/1.1 keep-alive를 위한 httpx 클라이언트 (동시성/지속연결 튜닝)
        http_client = AsyncClient(limits=Limits(max_connections=100, max_keepalive_connections=20))

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),  # Ray/vLLM는 보통 "EMPTY" 허용
            timeout=Timeout(timeout_s),
            max_retries=max_retries,  # ← 여기서 재시도 횟수 설정
            http_client=http_client,
        )
        self.model_name = model_name

        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def _canon_base_url(host: str) -> str:
        """host 문자열을 http(s) 스킴/슬래시/v1까지 표준화"""
        host = host.strip().rstrip("/")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        if not host.endswith("/v1"):
            host = f"{host}/v1"
        return host

    def log(self, message: str):
        self.logger.info(message)

    @staticmethod
    def load_queries(query_path: str) -> Tuple[List[Tuple[str, List[Dict[str, str]]]], List[Dict[str, Any]]]:
        queries, original_queries = [], []
        with jsonlines.open(query_path) as reader:
            for line in reader:
                if line["dedup_priority"] == 1:
                    _id = line["id"]
                    query_and_response = line["query_and_response"]
                    
                    # Convert "from" to "role" for OpenAI compatibility
                    messages = []
                    for msg in query_and_response:
                        # Skip the last message if it's from assistant with empty content
                        if (msg == query_and_response[-1] and 
                            msg.get("from") == "assistant" and 
                            not msg.get("content", "").strip()):
                            continue
                            
                        # Convert "from" to "role" for OpenAI compatibility
                        role = msg.get("from", msg.get("role", "user"))
                        content = msg.get("content", "")
                        
                        # Map role names to OpenAI standard
                        if role in ["human", "user"]:
                            role = "user"
                        elif role in ["gpt", "assistant", "assistance"]:
                            role = "assistant"
                        elif role == "system":
                            role = "system"
                        else:
                            role = "user"  # Default fallback
                            
                        messages.append({"role": role, "content": content})
                    
                    queries.append((_id, messages))
                    original_queries.append(line)
        return queries, original_queries

    async def generate_one(self, request_id: str, messages: List[Dict[str, str]], *args, **kwargs) -> str:
        out = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            extra_body={"chat_template_kwargs": {"thinking": True}},    # Guided from vLLM DeepSeek-V3.1 Guide (https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_1.html#introduction)
            # extra_headers={"x-request-id": request_id},  # 필요시 추적용 헤더
        )
        return out.choices[0].message.content or ""

    async def generate_batch(self,
                             prompts: List[Tuple[str, List[Dict[str, str]]]],
                             max_concurrent: int = 40,
                             desc: str = "mini_batch") -> List[Tuple[str, str]]:
        sem = asyncio.Semaphore(max_concurrent)
        results: List[Optional[Tuple[str, str]]] = [None] * len(prompts)

        async def _worker(i, request_id, messages):
            async with sem:
                try:
                    result = await self.generate_one(request_id, messages)
                    results[i] = (request_id, result)
                except Exception as e:
                    results[i] = (request_id, f"[ERROR] {type(e).__name__}: {e}")

        tasks = [asyncio.create_task(_worker(i, rid, msgs)) for i, (rid, msgs) in enumerate(prompts)]

        # asyncio.as_completed로 완료되는 순서대로 기다리면서 tqdm로 진행률 표시
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            await fut

        # type: ignore (위에서 모두 채워짐)
        return results  # type: ignore

    async def run(self, queries: List[Tuple[str, List[Dict[str, str]]]], original_queries: List[Dict[str, Any]], output_path: str, max_concurrent: int = 40):
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        mode = "a" if os.path.exists(output_path) else "w"
        self.log(f"Writing to {output_path} (mode={mode})")
        with jsonlines.open(output_path, mode=mode, flush=True) as writer:
            for begin in range(0, len(queries), max_concurrent):
                end = min(begin + max_concurrent, len(queries))
                prompts = [(queries[j][0], queries[j][1]) for j in range(begin, end)]
                results = await self.generate_batch(prompts, max_concurrent)

                # 결과 병합 (마지막 메시지 자리에 덮어쓰기, 없으면 append)
                for j, (_id, result) in enumerate(results):
                    record = original_queries[begin + j]
                    qrs = record.get("query_and_response") or []
                    
                    # Check if the last message is empty assistant message
                    if (qrs and isinstance(qrs[-1], dict) and 
                        qrs[-1].get("from") == "assistant" and 
                        not qrs[-1].get("content", "").strip()):
                        # Replace the empty assistant message with the result
                        qrs[-1]["content"] = result
                    else:
                        # Append new assistant message
                        qrs.append({"from": "assistant", "content": result})
                        record["query_and_response"] = qrs
                    original_queries[begin + j] = record

                writer.write_all(original_queries[begin:end])
                self.log(f"Generated {end}/{len(queries)} queries")

if __name__ == "__main__":
    import argparse, math
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1:8000")
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--timeout_s", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--max_concurrent", type=int, default=84)
    args = parser.parse_args()
    ratio = args.ratio
    import glob
    # 클러스터 내부 기본값: 헤드 서비스의 Serve 포트(8000)
    SERVE_HOST = os.getenv(
        "RAY_SERVE_HOST",
        args.host
    )
    generator = AsyncGenerator(
        model_name="deepseek-ai/DeepSeek-V3.1",   # 서버에 등록된 모델명과 정확히 일치
        host=f"http://{args.host}",              # ← Ray Serve 주소로!
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,                            # ← 재시도는 여기서 설정됨
    )
    jsonl_files = glob.glob("/data/datasets/original_lang_priority_divided/en/*.jsonl")
    total_iteration = math.ceil(1.0 / ratio)
    for i in range(total_iteration):
        for jsonl_file in jsonl_files:
            generator.log(f"Loading queries from {jsonl_file}")
            queries, original_queries = generator.load_queries(jsonl_file)
            n = len(queries)
            generator.log(f"Loaded {n} queries")
            
            # 각 iteration의 슬라이스 범위 계산
            if i < total_iteration - 1:
                start = math.floor(n * ratio * i)
                end = math.floor(n * ratio * (i + 1))
            else:
                # 마지막 iteration: 남은 전부 처리
                start = math.floor(n * ratio * i)
                end = n

            # 비어있는 경우 스킵
            if start >= n or end <= start:
                generator.log(f"Iteration {i}: empty slice (start={start}, end={end}), skipping")
                continue

            q_slice = queries[start:end]
            oq_slice = original_queries[start:end]
            generator.log(
                f"Iteration {i}: total={n}, ratio={ratio}, slice=[{start}:{end}] => {len(q_slice)} queries"
            )

            asyncio.run(generator.run(
                queries=q_slice,
                original_queries=oq_slice,
                output_path=jsonl_file.replace("en", "en_sample").replace(".jsonl", "_response_generated_sample.jsonl"),
                max_concurrent=args.max_concurrent,
            ))