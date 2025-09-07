import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from .file_manager import FileManager, FileMonitor
from .async_generator import AsyncGenerator
from .server_monitor import ServerHealthChecker


class GenerateResponseProcessor:
    """Generate Response 시스템의 메인 프로세서"""
    
    def __init__(self, 
                 model_name: str = "deepseek-ai/DeepSeek-V3.1",
                 server_url: str = "http://ray-deepseek-serve.p-ncai-wbl.svc.cluster.local:8000",
                 request_dir: str = "/data/data_team/request",
                 processing_dir: str = "/data/data_team/processing",
                 done_dir: str = "/data/data_team/done",
                 max_concurrent: int = 40,
                 timeout_s: float = 300.0,
                 max_retries: int = 5):
        
        self.model_name = model_name
        self.server_url = server_url
        self.max_concurrent = max_concurrent
        
        # 컴포넌트 초기화
        self.file_manager = FileManager(request_dir, processing_dir, done_dir)
        self.generator = AsyncGenerator(
            model_name=model_name,
            host=server_url,
            timeout_s=timeout_s,
            max_retries=max_retries,
            enable_server_monitoring=True
        )
        
        self.logger = logging.getLogger(__name__)
        self._running = False
        self._monitor = None
        
    async def process_single_file(self, file_path: Path) -> bool:
        """단일 파일을 처리합니다"""
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # 파일을 processing 디렉토리로 이동
            processing_path = self.file_manager.move_to_processing(file_path)
            
            # 출력 파일 경로 생성
            output_path = self.file_manager.done_dir / f"{processing_path.stem}_processed{processing_path.suffix}"
            
            # 파일 처리
            success = await self.generator.process_file(
                str(processing_path),
                str(output_path),
                self.max_concurrent
            )
            
            if success:
                # 처리 완료된 파일을 done 디렉토리로 이동
                self.file_manager.move_to_done(processing_path)
                self.logger.info(f"Successfully processed: {file_path.name}")
                return True
            else:
                self.logger.error(f"Failed to process: {file_path.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    async def process_existing_files(self):
        """기존에 있는 파일들을 처리합니다"""
        pending_files = self.file_manager.get_pending_files()
        
        if not pending_files:
            self.logger.info("No pending files to process")
            return
        
        self.logger.info(f"Found {len(pending_files)} pending files")
        
        for file_path in pending_files:
            if not self._running:
                break
                
            await self.process_single_file(file_path)
    
    async def handle_new_file(self, file_path: Path):
        """새로운 파일이 감지되었을 때 처리합니다"""
        if not self._running:
            return
            
        self.logger.info(f"New file detected: {file_path}")
        
        # 파일이 완전히 쓰여질 때까지 잠시 대기
        await asyncio.sleep(2)
        
        await self.process_single_file(file_path)
    
    async def start_monitoring(self):
        """파일 모니터링을 시작합니다"""
        self.logger.info("Starting Generate Response Processor")
        self._running = True

        # 서버 상태 확인
        self.logger.info("Checking server health...")
        if not await ServerHealthChecker.wait_for_server(self.server_url, max_wait_time=300.0):
            self.logger.error("Server is not healthy, cannot start processing")
            return False

        # 기존 파일들 처리
        await self.process_existing_files()

        # 파일 모니터링 시작 (주기적 검사 포함)
        self._monitor = FileMonitor(
            self.file_manager,
            callback=self.handle_new_file,
            check_interval=60  # 60초마다 주기적 검사
        )
        self._monitor.start_monitoring()

        self.logger.info("Generate Response Processor started successfully")
        return True
    
    async def stop_monitoring(self):
        """파일 모니터링을 중지합니다"""
        self.logger.info("Stopping Generate Response Processor")
        self._running = False
        
        if self._monitor:
            self._monitor.stop_monitoring()
            self._monitor = None
        
        self.logger.info("Generate Response Processor stopped")
    
    async def run_forever(self):
        """프로세서를 계속 실행합니다"""
        if not await self.start_monitoring():
            return
        
        try:
            while self._running:
                # 주기적으로 processing 디렉토리 정리
                self.file_manager.cleanup_processing()
                await asyncio.sleep(300)  # 5분마다 정리
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            await self.stop_monitoring()
    
    async def process_batch_once(self):
        """한 번만 배치 처리를 실행합니다 (스케줄러용)"""
        self.logger.info("Starting batch processing")
        self._running = True
        
        try:
            # 서버 상태 확인
            if not await ServerHealthChecker.wait_for_server(self.server_url, max_wait_time=60.0):
                self.logger.error("Server is not healthy, skipping batch processing")
                return False
            
            # 기존 파일들 처리
            await self.process_existing_files()
            
            # processing 디렉토리 정리
            self.file_manager.cleanup_processing()
            
            self.logger.info("Batch processing completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return False
        finally:
            self._running = False


async def main():
    """메인 실행 함수"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Generate Response Processor")
    parser.add_argument("--mode", choices=["monitor", "batch"], default="monitor",
                       help="실행 모드: monitor (지속 모니터링) 또는 batch (한 번만 실행)")
    parser.add_argument("--server-url", default="http://ray-deepseek-serve.p-ncai-wbl.svc.cluster.local:8000",
                       help="서버 URL")
    parser.add_argument("--model-name", default="deepseek-ai/DeepSeek-V3.1",
                       help="모델 이름")
    parser.add_argument("--max-concurrent", type=int, default=40,
                       help="최대 동시 요청 수")
    parser.add_argument("--timeout", type=float, default=300.0,
                       help="요청 타임아웃 (초)")
    parser.add_argument("--max-retries", type=int, default=5,
                       help="최대 재시도 횟수")
    parser.add_argument("--request-dir", default="/data/data_team/request",
                       help="요청 파일 디렉토리")
    parser.add_argument("--processing-dir", default="/data/data_team/processing",
                       help="처리 중 파일 디렉토리")
    parser.add_argument("--done-dir", default="/data/data_team/done",
                       help="완료된 파일 디렉토리")
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 프로세서 생성
    processor = GenerateResponseProcessor(
        model_name=args.model_name,
        server_url=args.server_url,
        request_dir=args.request_dir,
        processing_dir=args.processing_dir,
        done_dir=args.done_dir,
        max_concurrent=args.max_concurrent,
        timeout_s=args.timeout,
        max_retries=args.max_retries
    )
    
    # 실행 모드에 따라 처리
    if args.mode == "monitor":
        await processor.run_forever()
    else:  # batch
        success = await processor.process_batch_once()
        exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
