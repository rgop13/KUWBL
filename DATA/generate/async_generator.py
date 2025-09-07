import os
import sys
import asyncio
import logging
import jsonlines
from typing import List, Tuple, Any, Dict, Optional
from pathlib import Path

from openai import AsyncOpenAI
from httpx import Limits, Timeout, AsyncClient
from tqdm import tqdm

from .server_monitor import ServerMonitor, ServerHealthChecker
from .tool_agent import tool_agent_manager, ToolAgent


class DataProcessor:
    """데이터 포맷 처리 및 변환을 담당하는 클래스"""
    
    @staticmethod
    def load_queries(query_path: str) -> Tuple[List[Tuple[str, List[Dict[str, str]], Optional[List[Dict]]]], List[Dict[str, Any]]]:
        """JSONL 파일에서 쿼리를 로드하고 변환합니다"""
        queries, original_queries = [], []
        
        with jsonlines.open(query_path) as reader:
            for line in reader:
                # dedup_priority가 1인 것만 처리 (기존 로직 유지)
                if line.get("dedup_priority") == 1:
                    _id = line["id"]
                    query_and_response = line["query_and_response"]
                    tools = line.get("tools")  # tool 정보 추출
                    
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
                        
                        # tool_calls 정보도 포함
                        message = {"role": role, "content": content}
                        if "tool_calls" in msg:
                            message["tool_calls"] = msg["tool_calls"]
                        if "tool_call_id" in msg:
                            message["tool_call_id"] = msg["tool_call_id"]
                        if "name" in msg:
                            message["name"] = msg["name"]
                            
                        messages.append(message)
                    
                    queries.append((_id, messages, tools))
                    original_queries.append(line)
                    
        return queries, original_queries
    
    @staticmethod
    def is_tool_task(tools: Optional[List[Dict]]) -> bool:
        """tool task인지 판단합니다"""
        return tools is not None and len(tools) > 0
    
    @staticmethod
    def update_response(original_data: Dict[str, Any], response: str, model_name: str) -> Dict[str, Any]:
        """응답으로 원본 데이터를 업데이트합니다"""
        qrs = original_data.get("query_and_response", [])
        
        # Check if the last message is empty assistant message
        if (qrs and isinstance(qrs[-1], dict) and 
            qrs[-1].get("from") == "assistant" and 
            not qrs[-1].get("content", "").strip()):
            # Replace the empty assistant message with the result
            qrs[-1]["content"] = response
            qrs[-1]["provider"] = model_name
        else:
            # Append new assistant message
            qrs.append({
                "from": "assistant", 
                "content": response,
                "provider": model_name
            })
            
        original_data["query_and_response"] = qrs
        return original_data


class AsyncGenerator:
    """최적화된 비동기 요청 처리 클래스"""
    
    def __init__(self, 
                 model_name: str, 
                 host: str, 
                 timeout_s: float = 60.0, 
                 max_retries: int = 3,
                 enable_server_monitoring: bool = True):
        self.model_name = model_name
        self.base_url = self._canon_base_url(host)
        
        # HTTP/1.1 keep-alive를 위한 httpx 클라이언트 (동시성/지속연결 튜닝)
        http_client = AsyncClient(limits=Limits(max_connections=100, max_keepalive_connections=20))
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            timeout=Timeout(timeout_s),
            max_retries=max_retries,
            http_client=http_client,
        )
        
        # 서버 모니터링 설정
        if enable_server_monitoring:
            self.server_monitor = ServerMonitor(
                server_url=host,
                timeout=timeout_s,
                max_retries=max_retries
            )
        else:
            self.server_monitor = None
        
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
    
    async def generate_one(self,
                          request_id: str,
                          messages: List[Dict[str, str]],
                          tools: Optional[List[Dict]] = None) -> str:
        """단일 요청 생성"""

        async def _make_request():
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "extra_body": {"chat_template_kwargs": {"thinking": True}},
            }

            # tool이 있는 경우 추가
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""

        # Tool task인지 확인
        agent = tool_agent_manager.get_agent()
        if agent and agent.is_tool_task(messages, tools):
            self.log(f"Processing tool task: {request_id}")

            # Tool Agent를 통한 처리
            async def generator_func(msgs, tls):
                return await self._make_request_internal(msgs, tls)

            updated_messages, final_response = await tool_agent_manager.process_with_tools(
                messages, tools or [], generator_func
            )
            return final_response

        # 일반 요청 처리
        if self.server_monitor:
            return await self.server_monitor.wait_with_retry(_make_request)
        else:
            return await _make_request()

    async def _make_request_internal(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None) -> str:
        """내부 요청 처리 (Tool Agent에서 사용)"""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "extra_body": {"chat_template_kwargs": {"thinking": True}},
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
    
    async def generate_batch(self,
                           prompts: List[Tuple[str, List[Dict[str, str]], Optional[List[Dict]]]],
                           max_concurrent: int = 10,
                           desc: str = "batch_processing") -> List[Tuple[str, str]]:
        """배치 요청 처리"""
        sem = asyncio.Semaphore(max_concurrent)
        results: List[Optional[Tuple[str, str]]] = [None] * len(prompts)
        
        async def _worker(i, request_id, messages, tools):
            async with sem:
                try:
                    result = await self.generate_one(request_id, messages, tools)
                    results[i] = (request_id, result)
                except Exception as e:
                    self.logger.error(f"Error processing {request_id}: {e}")
                    results[i] = (request_id, f"[ERROR] {type(e).__name__}: {e}")
        
        tasks = [
            asyncio.create_task(_worker(i, rid, msgs, tools)) 
            for i, (rid, msgs, tools) in enumerate(prompts)
        ]
        
        # asyncio.as_completed로 완료되는 순서대로 기다리면서 tqdm로 진행률 표시
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            await fut
        
        return results  # type: ignore
    
    async def process_file(self, 
                          input_path: str, 
                          output_path: str, 
                          max_concurrent: int = 10) -> bool:
        """파일 전체를 처리합니다"""
        try:
            # 서버 상태 확인
            if self.server_monitor:
                self.log("Checking server health before processing...")
                if not await self.server_monitor.wait_for_healthy():
                    self.log("Server is not healthy, aborting processing")
                    return False
            
            # 데이터 로드
            self.log(f"Loading queries from {input_path}")
            queries, original_queries = DataProcessor.load_queries(input_path)
            n = len(queries)
            self.log(f"Loaded {n} queries")
            
            if n == 0:
                self.log("No queries to process")
                return True
            
            # 출력 디렉토리 생성
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            
            # 배치 처리
            results = await self.generate_batch(queries, max_concurrent, f"Processing {Path(input_path).name}")
            
            # 결과 저장
            self.log(f"Saving results to {output_path}")
            with jsonlines.open(output_path, mode="w", flush=True) as writer:
                for i, (request_id, response) in enumerate(results):
                    updated_data = DataProcessor.update_response(
                        original_queries[i], 
                        response, 
                        self.model_name
                    )
                    writer.write(updated_data)
            
            self.log(f"Successfully processed {n} queries")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing file {input_path}: {e}")
            return False


if __name__ == "__main__":
    # 테스트용 코드
    async def test_generator():
        generator = AsyncGenerator(
            model_name="deepseek-ai/DeepSeek-V3.1",
            host="http://ray-deepseek-serve.p-ncai-wbl.svc.cluster.local:8000",
            timeout_s=300.0,
            max_retries=3
        )
        
        # 테스트 파일 처리
        success = await generator.process_file(
            input_path="DATA/data_example.jsonl",
            output_path="DATA/test_output.jsonl",
            max_concurrent=10,
            enable_server_monitoring=False
        )
        
        print(f"Processing completed: {success}")
    
    asyncio.run(test_generator())
