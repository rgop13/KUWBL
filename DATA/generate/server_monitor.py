import asyncio
import aiohttp
import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ServerStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    status: ServerStatus
    response_time: Optional[float]
    error: Optional[str]
    timestamp: float


class ServerMonitor:
    """서버 상태 감지 및 대기 로직을 담당하는 클래스"""
    
    def __init__(self, 
                 server_url: str,
                 health_endpoint: str = "/v1/models",
                 check_interval: float = 30.0,
                 timeout: float = 10.0,
                 max_retries: int = 3,
                 backoff_factor: float = 2.0):
        self.server_url = server_url.rstrip('/')
        self.health_endpoint = health_endpoint
        self.check_interval = check_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
        self.logger = logging.getLogger(__name__)
        self._last_status = ServerStatus.UNKNOWN
        self._consecutive_failures = 0
        
    async def check_health(self) -> HealthCheckResult:
        """서버 상태를 확인합니다"""
        url = f"{self.server_url}{self.health_endpoint}"
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        # 응답 내용도 확인 (모델 목록이 있는지)
                        try:
                            data = await response.json()
                            if isinstance(data, dict) and 'data' in data:
                                self._consecutive_failures = 0
                                return HealthCheckResult(
                                    status=ServerStatus.HEALTHY,
                                    response_time=response_time,
                                    error=None,
                                    timestamp=time.time()
                                )
                        except Exception as e:
                            self.logger.warning(f"Failed to parse health check response: {e}")
                    
                    # 상태 코드가 200이 아니거나 응답 파싱 실패
                    self._consecutive_failures += 1
                    return HealthCheckResult(
                        status=ServerStatus.UNHEALTHY,
                        response_time=response_time,
                        error=f"HTTP {response.status}",
                        timestamp=time.time()
                    )
                    
        except asyncio.TimeoutError:
            self._consecutive_failures += 1
            return HealthCheckResult(
                status=ServerStatus.UNHEALTHY,
                response_time=None,
                error="Timeout",
                timestamp=time.time()
            )
        except Exception as e:
            self._consecutive_failures += 1
            return HealthCheckResult(
                status=ServerStatus.UNHEALTHY,
                response_time=None,
                error=str(e),
                timestamp=time.time()
            )
    
    async def wait_for_healthy(self, max_wait_time: float = 600.0) -> bool:
        """서버가 정상 상태가 될 때까지 대기합니다"""
        start_time = time.time()
        attempt = 0
        
        self.logger.info(f"Waiting for server to become healthy (max {max_wait_time}s)")
        
        while time.time() - start_time < max_wait_time:
            attempt += 1
            result = await self.check_health()
            
            if result.status == ServerStatus.HEALTHY:
                self.logger.info(f"Server is healthy after {attempt} attempts ({time.time() - start_time:.1f}s)")
                return True
            
            # 백오프 전략으로 대기 시간 조정
            wait_time = min(self.check_interval * (self.backoff_factor ** min(attempt - 1, 5)), 60.0)
            
            self.logger.warning(
                f"Server unhealthy (attempt {attempt}): {result.error}. "
                f"Waiting {wait_time:.1f}s before retry..."
            )
            
            await asyncio.sleep(wait_time)
        
        self.logger.error(f"Server did not become healthy within {max_wait_time}s")
        return False
    
    async def wait_with_retry(self, operation, max_retries: Optional[int] = None) -> Any:
        """서버 상태를 확인하면서 작업을 재시도합니다"""
        if max_retries is None:
            max_retries = self.max_retries
            
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # 서버 상태 확인
                health_result = await self.check_health()
                
                if health_result.status != ServerStatus.HEALTHY:
                    self.logger.warning(f"Server unhealthy before operation (attempt {attempt + 1})")
                    
                    # 서버가 정상 상태가 될 때까지 대기
                    if not await self.wait_for_healthy():
                        raise Exception("Server did not recover in time")
                
                # 작업 실행
                return await operation()
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    # 백오프 대기
                    wait_time = self.backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
                    
        # 모든 재시도 실패
        raise last_exception
    
    async def start_monitoring(self, callback=None):
        """지속적인 서버 모니터링을 시작합니다"""
        self.logger.info("Starting server monitoring")
        
        while True:
            try:
                result = await self.check_health()
                
                # 상태 변화 감지
                if result.status != self._last_status:
                    self.logger.info(f"Server status changed: {self._last_status.value} -> {result.status.value}")
                    self._last_status = result.status
                    
                    if callback:
                        await callback(result)
                
                # 연속 실패 횟수에 따른 경고
                if self._consecutive_failures > 0:
                    self.logger.warning(f"Consecutive failures: {self._consecutive_failures}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)


class ServerHealthChecker:
    """간단한 서버 상태 확인 유틸리티"""
    
    @staticmethod
    async def is_server_healthy(server_url: str, timeout: float = 5.0) -> bool:
        """서버가 정상인지 간단히 확인"""
        monitor = ServerMonitor(server_url, timeout=timeout)
        result = await monitor.check_health()
        return result.status == ServerStatus.HEALTHY
    
    @staticmethod
    async def wait_for_server(server_url: str, max_wait_time: float = 300.0) -> bool:
        """서버가 정상 상태가 될 때까지 대기"""
        monitor = ServerMonitor(server_url)
        return await monitor.wait_for_healthy(max_wait_time)


if __name__ == "__main__":
    # 테스트용 코드
    async def test_monitor():
        logging.basicConfig(level=logging.INFO)
        
        # serve_deepseek.yaml에서 설정된 서비스 URL
        server_url = "http://ray-deepseek-serve.p-ncai-wbl.svc.cluster.local:8000"
        
        monitor = ServerMonitor(server_url)
        
        # 서버 상태 확인
        result = await monitor.check_health()
        print(f"Server status: {result.status.value}")
        print(f"Response time: {result.response_time}")
        print(f"Error: {result.error}")
        
        # 서버가 정상 상태가 될 때까지 대기
        is_healthy = await monitor.wait_for_healthy(max_wait_time=60.0)
        print(f"Server became healthy: {is_healthy}")
    
    asyncio.run(test_monitor())
