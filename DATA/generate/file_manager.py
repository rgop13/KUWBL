import os
import shutil
import time
import logging
import hashlib
import asyncio
from pathlib import Path
from typing import List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class FileManager:
    """파일 감지 및 관리를 담당하는 클래스"""

    def __init__(self,
                 request_dir: str = "/data/data_team/request",
                 processing_dir: str = "/data/data_team/processing",
                 done_dir: str = "/data/data_team/done",
                 pod_id: Optional[str] = None,
                 total_pods: int = 1):
        self.request_dir = Path(request_dir)
        self.processing_dir = Path(processing_dir)
        self.done_dir = Path(done_dir)

        # 분산 처리를 위한 Pod 정보
        self.pod_id = pod_id or os.getenv('HOSTNAME', 'pod-0')
        self.total_pods = total_pods

        # 디렉토리 생성
        for dir_path in [self.request_dir, self.processing_dir, self.done_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        
    def should_process_file(self, file_path: Path) -> bool:
        """이 Pod가 해당 파일을 처리해야 하는지 결정 (분산 처리)"""
        if self.total_pods <= 1:
            return True

        # 파일명 해시를 기반으로 Pod 할당
        file_hash = hashlib.md5(file_path.name.encode()).hexdigest()
        assigned_pod = int(file_hash, 16) % self.total_pods
        current_pod_index = int(self.pod_id.split('-')[-1]) if '-' in self.pod_id else 0

        return assigned_pod == current_pod_index

    def get_pending_files(self) -> List[Path]:
        """request 디렉토리에서 이 Pod가 처리해야 할 .jsonl 파일들을 반환"""
        all_files = list(self.request_dir.glob("*.jsonl"))
        return [f for f in all_files if self.should_process_file(f)]
    
    def move_to_processing(self, file_path: Path) -> Path:
        """파일을 request에서 processing으로 이동"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        dest_path = self.processing_dir / file_path.name
        
        # 이미 processing에 같은 이름의 파일이 있으면 타임스탬프 추가
        if dest_path.exists():
            timestamp = int(time.time())
            stem = dest_path.stem
            suffix = dest_path.suffix
            dest_path = self.processing_dir / f"{stem}_{timestamp}{suffix}"
            
        shutil.move(str(file_path), str(dest_path))
        self.logger.info(f"Moved {file_path} -> {dest_path}")
        return dest_path
    
    def move_to_done(self, file_path: Path) -> Path:
        """파일을 processing에서 done으로 이동"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        dest_path = self.done_dir / file_path.name
        
        # 이미 done에 같은 이름의 파일이 있으면 타임스탬프 추가
        if dest_path.exists():
            timestamp = int(time.time())
            stem = dest_path.stem
            suffix = dest_path.suffix
            dest_path = self.done_dir / f"{stem}_{timestamp}{suffix}"
            
        shutil.move(str(file_path), str(dest_path))
        self.logger.info(f"Moved {file_path} -> {dest_path}")
        return dest_path
    
    def cleanup_processing(self, max_age_hours: int = 24):
        """processing 디렉토리에서 오래된 파일들을 정리"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in self.processing_dir.glob("*.jsonl"):
            if current_time - file_path.stat().st_mtime > max_age_seconds:
                self.logger.warning(f"Cleaning up old processing file: {file_path}")
                # done으로 이동하거나 삭제
                try:
                    self.move_to_done(file_path)
                except Exception as e:
                    self.logger.error(f"Failed to cleanup {file_path}: {e}")


class FileWatcher(FileSystemEventHandler):
    """파일 시스템 이벤트를 감지하는 핸들러"""
    
    def __init__(self, callback=None):
        self.callback = callback
        self.logger = logging.getLogger(__name__)
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.jsonl'):
            self.logger.info(f"New file detected: {event.src_path}")
            if self.callback:
                self.callback(Path(event.src_path))
                
    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith('.jsonl'):
            self.logger.info(f"File moved to: {event.dest_path}")
            if self.callback:
                self.callback(Path(event.dest_path))


class FileMonitor:
    """파일 감지 및 처리를 위한 모니터"""

    def __init__(self, file_manager: FileManager, callback=None, check_interval: int = 60):
        self.file_manager = file_manager
        self.callback = callback
        self.observer = Observer()
        self.logger = logging.getLogger(__name__)
        self.check_interval = check_interval  # 주기적 검사 간격 (초)
        self._running = False
        self._periodic_task = None
        self._processed_files: Set[str] = set()  # 이미 처리된 파일들 추적

    def start_monitoring(self):
        """파일 모니터링 시작"""
        self._running = True

        # Watchdog 이벤트 기반 모니터링 시작
        handler = FileWatcher(callback=self._handle_new_file)
        self.observer.schedule(handler, str(self.file_manager.request_dir), recursive=False)
        self.observer.start()
        self.logger.info(f"Started event-based monitoring {self.file_manager.request_dir}")

        # 기존 파일들 처리
        for file_path in self.file_manager.get_pending_files():
            self._handle_new_file(file_path)

        # 주기적 검사 태스크 시작
        self._periodic_task = asyncio.create_task(self._periodic_check())
        self.logger.info(f"Started periodic checking every {self.check_interval} seconds")

    def stop_monitoring(self):
        """파일 모니터링 중지"""
        self._running = False

        # 주기적 검사 태스크 중지
        if self._periodic_task and not self._periodic_task.done():
            self._periodic_task.cancel()

        # Watchdog 모니터링 중지
        self.observer.stop()
        self.observer.join()
        self.logger.info("Stopped file monitoring")

    async def _periodic_check(self):
        """주기적으로 request 디렉토리를 검사하여 새 파일을 찾음"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)

                if not self._running:
                    break

                self.logger.debug("Performing periodic file check...")
                pending_files = self.file_manager.get_pending_files()

                for file_path in pending_files:
                    file_key = str(file_path)

                    # 이미 처리된 파일은 건너뛰기
                    if file_key not in self._processed_files:
                        self.logger.info(f"Periodic check found new file: {file_path}")
                        self._handle_new_file(file_path)

            except asyncio.CancelledError:
                self.logger.info("Periodic check task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in periodic check: {e}")

    def _handle_new_file(self, file_path: Path):
        """새 파일 처리"""
        try:
            file_key = str(file_path)

            # 이미 처리된 파일인지 확인
            if file_key in self._processed_files:
                return

            # 파일이 완전히 쓰여질 때까지 잠시 대기
            time.sleep(1)

            # 파일이 실제로 존재하는지 확인
            if not file_path.exists():
                self.logger.warning(f"File no longer exists: {file_path}")
                return

            # 처리된 파일로 마킹
            self._processed_files.add(file_key)

            if self.callback:
                self.callback(file_path)
            else:
                self.logger.info(f"New file ready for processing: {file_path}")

        except Exception as e:
            self.logger.error(f"Error handling new file {file_path}: {e}")


if __name__ == "__main__":
    # 테스트용 코드
    async def test_monitor():
        logging.basicConfig(level=logging.INFO)

        def test_callback(file_path):
            print(f"Processing file: {file_path}")

        file_manager = FileManager()
        monitor = FileMonitor(file_manager, callback=test_callback, check_interval=30)

        try:
            monitor.start_monitoring()
            print("File monitoring started (with periodic checking). Press Ctrl+C to stop.")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("File monitoring stopped.")

    asyncio.run(test_monitor())
