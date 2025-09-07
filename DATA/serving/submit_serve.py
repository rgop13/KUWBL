#!/usr/bin/env python3
"""
Ray Submit을 사용하여 Ray Serve를 클러스터 내부에서 실행
이 방법이 가장 안정적이고 권장되는 방식입니다.
"""

import os
import ray
import time
import argparse
from pathlib import Path

def submit_ray_serve():
    """Ray Submit을 사용하여 Ray Serve 배포"""
    
    # Ray 클러스터 연결
    ray_address = os.getenv("RAY_ADDRESS", "ray-ku-junyoung-head-0.ray-ku-junyoung-head.p-ncai-wbl.svc.cluster.local:6379")
    
    print(f"Connecting to Ray cluster: {ray_address}")
    
    # Ray Client 모드 비활성화
    os.environ["RAY_CLIENT_MODE"] = "0"
    
    try:
        ray.init(address=f"ray://{ray_address}", ignore_reinit_error=True)
        print("✓ Connected to Ray cluster")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False
    
    # Ray Job 제출
    job_config = {
        "runtime_env": {
            "working_dir": "/opt/WBL",
            "pip": ["vllm", "transformers", "torch"],
            "env_vars": {
                "MODEL_NAME": os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3.1"),
                "NUM_REPLICAS": os.getenv("NUM_REPLICAS", "16"),
                "TENSOR_PARALLEL_SIZE": os.getenv("TENSOR_PARALLEL_SIZE", "8"),
                "PIPELINE_PARALLEL_SIZE": os.getenv("PIPELINE_PARALLEL_SIZE", "2"),
                "MAX_MODEL_LEN": os.getenv("MAX_MODEL_LEN", "65536"),
                "GPU_MEMORY_UTILIZATION": os.getenv("GPU_MEMORY_UTILIZATION", "0.90"),
                "SERVING_PORT": os.getenv("SERVING_PORT", "8000"),
                "HF_HOME": "/data/data_team/cache/huggingface",
                "VLLM_USE_V1": "1",
            }
        }
    }
    
    try:
        # Ray Job 제출
        from ray.job_submission import JobSubmissionClient
        
        client = JobSubmissionClient(f"http://{ray_address.replace(':6379', ':8265')}")
        
        job_id = client.submit_job(
            entrypoint="python DATA/serving/serve_head_api.py --model_name=${MODEL_NAME} --num_replicas=${NUM_REPLICAS} --tensor_parallel_size=${TENSOR_PARALLEL_SIZE} --pipeline_parallel_size=${PIPELINE_PARALLEL_SIZE}",
            runtime_env=job_config["runtime_env"]
        )
        
        print(f"✓ Ray Job submitted: {job_id}")
        
        # Job 상태 모니터링
        while True:
            status = client.get_job_status(job_id)
            print(f"Job status: {status}")
            
            if status in ["SUCCEEDED", "FAILED", "STOPPED"]:
                break
                
            time.sleep(10)
        
        if status == "SUCCEEDED":
            print("✓ Ray Serve started successfully")
            return True
        else:
            print(f"❌ Job failed with status: {status}")
            logs = client.get_job_logs(job_id)
            print("Job logs:")
            print(logs)
            return False
            
    except Exception as e:
        print(f"Job submission failed: {e}")
        return False
    
    finally:
        ray.shutdown()

def main():
    parser = argparse.ArgumentParser(description="Submit Ray Serve job")
    parser.add_argument("--wait", action="store_true", help="Wait for job completion")
    args = parser.parse_args()
    
    success = submit_ray_serve()
    
    if success:
        print("Ray Serve deployment completed successfully")
        if args.wait:
            print("Keeping process alive...")
            try:
                while True:
                    time.sleep(60)
            except KeyboardInterrupt:
                print("Shutting down...")
    else:
        print("Ray Serve deployment failed")
        exit(1)

if __name__ == "__main__":
    main()
