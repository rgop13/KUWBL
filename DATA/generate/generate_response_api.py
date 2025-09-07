import os
import sys
import asyncio
import logging
import jsonlines
from typing import List, Tuple, Any, Dict, Optional

from openai import AsyncOpenAI
from httpx import Limits, Timeout, AsyncClient
from tqdm import tqdm  # 진행률 바

from together import Together, AsyncTogether

class AsyncGenerator:
    def __init__(self, model_name: str, host: str, *, timeout_s: float = 60.0, max_retries: int = 2, api_key: str):
        # base_url = self._canon_base_url(host)
        os.environ["TOGETHER_API_KEY"] = api_key

        # HTTP/1.1 keep-alive를 위한 httpx 클라이언트 (동시성/지속연결 튜닝)
        # http_client = AsyncClient(limits=Limits(max_connections=100, max_keepalive_connections=20))
        self.client = AsyncTogether()
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
        # if not host.endswith("/v1"):
        #     host = f"{host}/v1"
        return host

    def log(self, message: str):
        self.logger.info(message)

    @staticmethod
    def load_queries(query_path: str) -> Tuple[List[Tuple[str, str, Optional[List[Dict]]]], List[Dict[str, Any]]]:
        queries, original_queries = [], []
        try:
            with jsonlines.open(query_path) as reader:
                for line_num, line in enumerate(reader, 1):
                    try:
                        if line["dedup_priority"] == 1:
                            _id = line["id"]
                            _query = line["query_and_response"][0]["content"]
                            
                            # Extract tools from system message if present
                            tools = None
                            query_and_response = line.get("query_and_response", [])
                            
                            # Look for system message with tool_calls
                            for message in query_and_response:
                                if message.get("from") == "system" and "tool_calls" in message:
                                    tools = []
                                    for tool_call in message["tool_calls"]:
                                        if tool_call.get("type") == "function" and "function" in tool_call:
                                            function_info = tool_call["function"]
                                            
                                            # Skip malformed entries that are actually parameters, not functions
                                            if "parameter_name" in function_info or ("name" not in function_info and "type" in function_info and "description" in function_info):
                                                print(f"DEBUG: Line {line_num}, skipping malformed function entry (appears to be a parameter): {function_info}")
                                                continue
                                                
                                            # Debug: print the structure if name is missing
                                            if "name" not in function_info:
                                                print(f"DEBUG: Line {line_num}, function_info missing 'name': {function_info}")
                                                print(f"DEBUG: tool_call structure: {tool_call}")
                                                continue
                                            
                                            tool_def = {
                                                "type": "function",
                                                "function": {
                                                    "name": function_info["name"],
                                                    "description": function_info.get("description", ""),
                                                    "parameters": {
                                                        "type": "object",
                                                        "properties": {},
                                                        "required": []
                                                    }
                                                }
                                            }
                                            
                                            # Handle parameters
                                            if "parameters" in function_info:
                                                for param_name, param_info in function_info["parameters"].items():
                                                    # Handle case where param_info might be a string or dict
                                                    if isinstance(param_info, dict):
                                                        tool_def["function"]["parameters"]["properties"][param_name] = {
                                                            "type": param_info.get("type", "string"),
                                                            "description": param_info.get("description", "")
                                                        }
                                                        if param_info.get("default") is not None:
                                                            tool_def["function"]["parameters"]["properties"][param_name]["default"] = param_info["default"]
                                                    else:
                                                        # If param_info is a string, treat it as the type
                                                        tool_def["function"]["parameters"]["properties"][param_name] = {
                                                            "type": str(param_info) if param_info else "string",
                                                            "description": ""
                                                        }
                                            
                                            tools.append(tool_def)
                                    break
                            
                            queries.append((_id, _query, tools))
                            original_queries.append(line)
                    except Exception as e:
                        print(f"ERROR: Line {line_num} in {query_path}: {e}")
                        print(f"Problematic line: {line}")
                        continue
        except Exception as e:
            print(f"ERROR: Failed to read {query_path}: {e}")
            raise
        return queries, original_queries

    async def generate_one(self, request_id: str, prompt: str, tools: Optional[List[Dict]] = None) -> Tuple[str, Optional[Dict]]:
        safety_prompt = """Default to thorough, practical help. If content is sensitive, give high-level frameworks and safer alternatives with minimal hedging. If dangerous/illicit, refuse and redirect. Stay neutral—no moral judgments; keep disclaimers short; be as helpful as safely possible."""
        kwargs = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": safety_prompt},
                {"role": "user", "content": prompt}
            ],
            "chat_template_kwargs": {"thinking": True},
            "stream": False,
            "max_tokens": 65536,
        }
        
        # Add tools and tool_choice if tools are present
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        out = await self.client.chat.completions.create(**kwargs)
        
        message = out.choices[0].message
        result = ""
        tool_info = None
        
        # Add reasoning if available
        if hasattr(message, 'reasoning') and message.reasoning:
            result += "<think> " + message.reasoning + " </think> "
        
        # Add content if available
        if message.content:
            result += message.content
        
        # Extract tool information separately
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_info = {
                "tool_calls": [],
                "finish_reason": out.choices[0].finish_reason
            }
            for tool_call in message.tool_calls:
                tool_info["tool_calls"].append({
                    "id": getattr(tool_call, 'id', None),
                    "type": getattr(tool_call, 'type', 'function'),
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                })
        
        # Handle legacy function call format
        elif hasattr(message, 'function_call') and message.function_call:
            tool_info = {
                "function_call": {
                    "name": message.function_call.name,
                    "arguments": message.function_call.arguments
                },
                "finish_reason": out.choices[0].finish_reason
            }
        
        # Add finish reason if it's not normal completion
        if not tool_info and out.choices[0].finish_reason != "stop":
            tool_info = {"finish_reason": out.choices[0].finish_reason}
        
        return result if result else "[No content generated]", tool_info

    async def generate_batch(self,
                             prompts: List[Tuple[str, str, Optional[List[Dict]]]],
                             max_concurrent: int = 40,
                             desc: str = "mini_batch") -> List[Tuple[str, str, Optional[Dict]]]:
        sem = asyncio.Semaphore(max_concurrent)
        results: List[Optional[Tuple[str, str, Optional[Dict]]]] = [None] * len(prompts)

        async def _worker(i, request_id, prompt, tools):
            async with sem:
                try:
                    result, tool_info = await self.generate_one(request_id, prompt, tools)
                    results[i] = (request_id, result, tool_info)
                except Exception as e:
                    results[i] = (request_id, f"[ERROR] {type(e).__name__}: {e}", None)

        tasks = [asyncio.create_task(_worker(i, rid, pmt, tools)) for i, (rid, pmt, tools) in enumerate(prompts)]

        # asyncio.as_completed로 완료되는 순서대로 기다리면서 tqdm로 진행률 표시
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            await fut

        # type: ignore (위에서 모두 채워짐)
        return results  # type: ignore

    async def run(self, queries: List[Tuple[str, str, Optional[List[Dict]]]], original_queries: List[Dict[str, Any]], output_path: str, max_concurrent: int = 40):
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        mode = "a" if os.path.exists(output_path) else "w"
        self.log(f"Writing to {output_path} (mode={mode})")
        with jsonlines.open(output_path, mode=mode, flush=True) as writer:
            for begin in range(0, len(queries), max_concurrent):
                end = min(begin + max_concurrent, len(queries))
                prompts = [(queries[j][0], queries[j][1], queries[j][2]) for j in range(begin, end)]
                results = await self.generate_batch(prompts, max_concurrent)

                # 결과 병합 (마지막 메시지 자리에 덮어쓰기, 없으면 append)
                for j, (_id, result, tool_info) in enumerate(results):
                    record = original_queries[begin + j]
                    qrs = record.get("query_and_response") or []
                    
                    # Create assistant message
                    assistant_msg = {"role": "assistant", "content": result}
                    
                    # Add tool information if present
                    if tool_info:
                        if "tool_calls" in tool_info:
                            assistant_msg["tool_calls"] = tool_info["tool_calls"]
                        if "function_call" in tool_info:
                            assistant_msg["function_call"] = tool_info["function_call"]
                        if "finish_reason" in tool_info:
                            assistant_msg["finish_reason"] = tool_info["finish_reason"]
                    
                    if qrs and isinstance(qrs[-1], dict) and "content" in qrs[-1]:
                        # Update existing assistant message
                        qrs[-1] = assistant_msg
                    else:
                        # Add new assistant message
                        qrs.append(assistant_msg)
                        record["query_and_response"] = qrs
                    original_queries[begin + j] = record

                writer.write_all(original_queries[begin:end])
                self.log(f"Generated {end}/{len(queries)} queries")
                await asyncio.sleep(1)

if __name__ == "__main__":
    import argparse, math
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="https://api.deepseek.com")
    parser.add_argument("--ratio", type=float, default=1.00)
    parser.add_argument("--timeout_s", type=float, default=240.0)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--max_concurrent", type=int, default=10)
    args = parser.parse_args()
    ratio = args.ratio
    with open("secret/deepseek.key", "r") as f:
        api_key = f.read().strip()
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
        api_key=api_key,
    )
    
    jsonl_files = glob.glob("/data/datasets/sampled_data/safety_*.jsonl")
    if not jsonl_files:
        # Fallback to alternative location if main path doesn't exist
        jsonl_files = glob.glob("/data_x/WBL/data/sampled_data/safety_*.jsonl")
    total_iteration = math.ceil(1.0 / ratio)
    for i in range(total_iteration):
        for jsonl_file in jsonl_files:
            file_name = os.path.basename(jsonl_file)
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
            if len(q_slice) > 500:
                q_slice = q_slice[:500]
                oq_slice = oq_slice[:500]
            generator.log(
                f"Iteration {i}: total={n}, ratio={ratio}, slice=[{start}:{end}] => {len(q_slice)} queries"
            )
            base_dir = os.path.dirname(jsonl_file)
            file_name = os.path.basename(jsonl_file)
            output_path = os.path.join(base_dir, "generated")
            if not os.path.exists(output_path):
                os.makedirs(output_path, exist_ok=True)
            
            asyncio.run(generator.run(
                queries=q_slice,
                original_queries=oq_slice,
                output_path=os.path.join(output_path, file_name.replace(".jsonl", "_response_generated_3.jsonl")),
                max_concurrent=args.max_concurrent,
            ))
        print("5% processing is Done!")
        break