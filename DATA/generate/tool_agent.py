import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Tool 호출 정보"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResult:
    """Tool 실행 결과"""
    tool_call_id: str
    name: str
    content: str
    success: bool
    error: Optional[str] = None


class ToolExecutor(ABC):
    """Tool 실행을 위한 추상 클래스"""
    
    @abstractmethod
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Tool을 실행하고 결과를 반환합니다"""
        pass
    
    @abstractmethod
    def get_supported_tools(self) -> List[str]:
        """지원하는 tool 목록을 반환합니다"""
        pass


class MockToolExecutor(ToolExecutor):
    """테스트용 Mock Tool Executor"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Mock tool 실행 - 실제 구현에서는 각 tool에 맞게 구현해야 함"""
        self.logger.info(f"Executing mock tool: {tool_call.name} with args: {tool_call.arguments}")
        
        # 간단한 mock 응답 생성
        mock_responses = {
            "get_weather": f"The weather is sunny with temperature 25°C",
            "calculate": f"The calculation result is 42",
            "search": f"Search results for your query",
            "get_news_headlines": json.dumps({
                "headlines": ["Mock headline 1", "Mock headline 2", "Mock headline 3"]
            }),
            "convert_temperature": json.dumps({
                "converted_temperature": 86
            }),
            "create_contact": "Contact created successfully",
            "generate_password": "Generated password: MockPass123!",
            "calculate_median": json.dumps({"median": 5.5}),
            "calculate_loan_payment": json.dumps({"monthly_payment": 530.33})
        }
        
        content = mock_responses.get(tool_call.name, f"Mock result for {tool_call.name}")
        
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content=content,
            success=True
        )
    
    def get_supported_tools(self) -> List[str]:
        return [
            "get_weather", "calculate", "search", "get_news_headlines",
            "convert_temperature", "create_contact", "generate_password",
            "calculate_median", "calculate_loan_payment"
        ]


class ToolAgent:
    """Tool Agent 관리자 - tool selection, 실행, 재요청을 관리합니다"""
    
    def __init__(self, tool_executor: ToolExecutor):
        self.tool_executor = tool_executor
        self.logger = logging.getLogger(__name__)
        
    def extract_tool_calls(self, assistant_message: Dict[str, Any]) -> List[ToolCall]:
        """Assistant 메시지에서 tool_calls를 추출합니다"""
        tool_calls = []
        
        if "tool_calls" in assistant_message:
            for tc in assistant_message["tool_calls"]:
                if tc.get("type") == "function":
                    function = tc.get("function", {})
                    try:
                        arguments = json.loads(function.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        arguments = {}
                    
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=function.get("name", ""),
                        arguments=arguments
                    ))
        
        return tool_calls
    
    async def execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Tool들을 실행하고 결과를 반환합니다"""
        results = []
        
        for tool_call in tool_calls:
            try:
                result = await self.tool_executor.execute_tool(tool_call)
                results.append(result)
                self.logger.info(f"Tool {tool_call.name} executed successfully")
            except Exception as e:
                self.logger.error(f"Error executing tool {tool_call.name}: {e}")
                results.append(ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error: {str(e)}",
                    success=False,
                    error=str(e)
                ))
        
        return results
    
    def create_tool_messages(self, tool_results: List[ToolResult]) -> List[Dict[str, Any]]:
        """Tool 실행 결과를 메시지 형태로 변환합니다"""
        messages = []
        
        for result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": result.tool_call_id,
                "name": result.name,
                "content": result.content
            })
        
        return messages
    
    async def process_tool_conversation(self, 
                                      messages: List[Dict[str, Any]], 
                                      tools: List[Dict[str, Any]],
                                      generator_func) -> Tuple[List[Dict[str, Any]], str]:
        """
        Tool을 사용하는 대화를 처리합니다
        
        Args:
            messages: 현재까지의 메시지 목록
            tools: 사용 가능한 tool 정의
            generator_func: 모델 생성 함수 (async callable)
            
        Returns:
            (updated_messages, final_response): 업데이트된 메시지와 최종 응답
        """
        max_iterations = 5  # 무한 루프 방지
        current_messages = messages.copy()
        
        for iteration in range(max_iterations):
            self.logger.info(f"Tool conversation iteration {iteration + 1}")
            
            # 모델에 요청
            response = await generator_func(current_messages, tools)
            
            # 응답을 메시지에 추가
            assistant_message = {
                "role": "assistant",
                "content": response
            }
            
            # Tool calls가 있는지 확인 (실제로는 모델 응답에서 파싱해야 함)
            # 여기서는 간단히 구현 - 실제로는 모델 응답을 파싱해야 함
            tool_calls = self.extract_tool_calls(assistant_message)
            
            if not tool_calls:
                # Tool call이 없으면 대화 완료
                current_messages.append(assistant_message)
                return current_messages, response
            
            # Tool calls가 있으면 실행
            assistant_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
            current_messages.append(assistant_message)
            
            # Tool 실행
            tool_results = await self.execute_tools(tool_calls)
            
            # Tool 결과를 메시지에 추가
            tool_messages = self.create_tool_messages(tool_results)
            current_messages.extend(tool_messages)
            
            self.logger.info(f"Executed {len(tool_calls)} tools, continuing conversation...")
        
        # 최대 반복 횟수 도달
        self.logger.warning(f"Reached maximum iterations ({max_iterations}) for tool conversation")
        return current_messages, "Tool conversation reached maximum iterations"
    
    def is_tool_task(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> bool:
        """주어진 메시지와 tools가 tool task인지 판단합니다"""
        # tools가 정의되어 있으면 tool task
        if tools and len(tools) > 0:
            return True
        
        # 메시지에 tool 관련 내용이 있으면 tool task
        for message in messages:
            if message.get("role") == "tool" or "tool_calls" in message:
                return True
        
        return False


class ToolAgentManager:
    """Tool Agent들을 관리하는 매니저 클래스"""
    
    def __init__(self):
        self.agents: Dict[str, ToolAgent] = {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 Mock Tool Agent 등록
        mock_executor = MockToolExecutor()
        self.register_agent("default", ToolAgent(mock_executor))
    
    def register_agent(self, name: str, agent: ToolAgent):
        """Tool Agent를 등록합니다"""
        self.agents[name] = agent
        self.logger.info(f"Registered tool agent: {name}")
    
    def get_agent(self, name: str = "default") -> Optional[ToolAgent]:
        """Tool Agent를 가져옵니다"""
        return self.agents.get(name)
    
    async def process_with_tools(self, 
                               messages: List[Dict[str, Any]], 
                               tools: List[Dict[str, Any]],
                               generator_func,
                               agent_name: str = "default") -> Tuple[List[Dict[str, Any]], str]:
        """Tool을 사용하여 대화를 처리합니다"""
        agent = self.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Tool agent '{agent_name}' not found")
        
        return await agent.process_tool_conversation(messages, tools, generator_func)


# 전역 Tool Agent Manager 인스턴스
tool_agent_manager = ToolAgentManager()


if __name__ == "__main__":
    # 테스트용 코드
    import asyncio
    
    async def test_tool_agent():
        logging.basicConfig(level=logging.INFO)
        
        # Mock generator function
        async def mock_generator(messages, tools):
            return "This is a mock response that would normally come from the model"
        
        # 테스트 메시지
        messages = [
            {"role": "user", "content": "What's the weather like today?"}
        ]
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
        # Tool Agent 테스트
        manager = ToolAgentManager()
        updated_messages, response = await manager.process_with_tools(
            messages, tools, mock_generator
        )
        
        print(f"Final response: {response}")
        print(f"Message count: {len(updated_messages)}")
    
    asyncio.run(test_tool_agent())
