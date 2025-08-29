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

        # ⚠️ 재시도는 여기!
        # - max_retries: OpenAI 클라이언트의 HTTP 자동 재시도 횟수 (408/429/5xx 등에서 동작)
        # - timeout: 요청 타임아웃 (연결/읽기/전체)
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

    def load_queries(self, query_path: str) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
        queries, original_queries = [], []
        with jsonlines.open(query_path) as reader:
            for line in reader:
                if line["dedup_priority"] == 1:
                    _id = line["id"]
                    _query = line["query_and_response"][0]["content"]
                    queries.append((_id, _query))
                    original_queries.append(line)
        return queries, original_queries

    async def generate_one(self, request_id: str, prompt: str, *,
                           temperature: float = 0.2,
                           max_tokens: Optional[int] = 1024) -> str:
        out = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            # extra_headers={"x-request-id": request_id},  # 필요시 추적용 헤더
        )
        return out.choices[0].message.content or ""

    async def generate_batch(self,
                             prompts: List[Tuple[str, str]],
                             max_concurrent: int = 40,
                             desc: str = "mini_batch") -> List[Tuple[str, str]]:
        sem = asyncio.Semaphore(max_concurrent)
        results: List[Optional[Tuple[str, str]]] = [None] * len(prompts)

        async def _worker(i, request_id, prompt):
            async with sem:
                try:
                    result = await self.generate_one(request_id, prompt)
                    results[i] = (request_id, result)
                except Exception as e:
                    results[i] = (request_id, f"[ERROR] {type(e).__name__}: {e}")

        tasks = [asyncio.create_task(_worker(i, rid, pmt)) for i, (rid, pmt) in enumerate(prompts)]

        # asyncio.as_completed로 완료되는 순서대로 기다리면서 tqdm로 진행률 표시
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            await fut

        # type: ignore (위에서 모두 채워짐)
        return results  # type: ignore

    async def run(self, query_path: str, output_path: str, max_concurrent: int = 40):
        self.log(f"Loading queries from {query_path}")
        queries, original_queries = self.load_queries(query_path)
        self.log(f"Total queries: {len(queries)}")

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
                    if qrs and isinstance(qrs[-1], dict) and "content" in qrs[-1]:
                        qrs[-1]["content"] = result
                    else:
                        qrs.append({"role": "assistant", "content": result})
                        record["query_and_response"] = qrs
                    original_queries[begin + j] = record

                writer.write_all(original_queries[begin:end])
                self.log(f"Generated {end - begin}/{len(queries)} queries")

if __name__ == "__main__":
    # 클러스터 내부 기본값: 헤드 서비스의 Serve 포트(8000)
    SERVE_HOST = os.getenv(
        "RAY_SERVE_HOST",
        "ray-ku-head.p-ncai-wbl.svc.cluster.local:8000"
    )

    generator = AsyncGenerator(
        model_name="deepseek-ai/DeepSeek-V3.1",   # 서버에 등록된 모델명과 정확히 일치
        host=f"http://{SERVE_HOST}",              # ← Ray Serve 주소로!
        timeout_s=120.0,
        max_retries=5,                            # ← 재시도는 여기서 설정됨
    )
    asyncio.run(generator.run(
        query_path="/data/dedup_dataset/if_merged_datasets_30108.jsonl",
        output_path="/data/dedup_generated/if_merged_datasets_30108_generated.jsonl",
        max_concurrent=100,
    ))
