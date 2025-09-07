# 🚀 DeepSeek V3.1 최적화 가이드 (TP=16, PP=2)

## 📊 최적 설정 분석

### 🎯 **권장 구성**
```yaml
TENSOR_PARALLEL_SIZE: 16    # 16 GPU per replica
PIPELINE_PARALLEL_SIZE: 2   # 2-stage pipeline  
NUM_REPLICAS: 3             # 최적 replica 수
Total GPU: 96 (16 × 2 × 3)
```

### 🔄 **성능 vs 리소스 트레이드오프**

#### **Option 1: TP=16, PP=2, Replicas=3 (권장)**
- ✅ **GPU**: 96개 사용
- ✅ **처리량**: 높음 (3개 독립 replica)
- ✅ **안정성**: 최고 (장애 격리)
- ✅ **지연시간**: 낮음 (PP=2)
- ❌ **메모리**: 모델 3배 중복

#### **Option 2: TP=32, PP=1, Replicas=3**
- ✅ **GPU**: 96개 사용
- ✅ **지연시간**: 최저 (PP=1)
- ❌ **통신**: TP=32로 통신 오버헤드 증가
- ❌ **안정성**: 32 GPU 중 하나 실패시 전체 영향

#### **Option 3: TP=16, PP=2, Replicas=6**
- ✅ **처리량**: 최고
- ❌ **GPU**: 192개 필요
- ❌ **리소스**: 과도한 GPU 사용

## 🛡️ 안정성 개선사항

### **1. Pod 생명주기 최적화**
```yaml
# 무거운 모델 고려한 여유있는 설정
startupProbe:
  failureThreshold: 120    # 60분 허용
  periodSeconds: 30
  
livenessProbe:
  initialDelaySeconds: 600 # 10분 후 시작
  failureThreshold: 5      # 5분 여유
  
readinessProbe:
  initialDelaySeconds: 180 # 3분 초기화
  successThreshold: 2      # 안정성 확인
```

### **2. 리소스 여유 확보**
```yaml
resources:
  limits:
    cpu: "16"
    memory: "128Gi"        # 기존 64Gi → 128Gi
    ephemeral-storage: "50Gi"
  requests:
    cpu: "12"              # 최소 보장
    memory: "96Gi"         # 최소 보장
```

### **3. 통신 안정성 강화**
```yaml
# NCCL 최적화 (TP=16 고려)
NCCL_SOCKET_NTHREADS: "4"
NCCL_NSOCKS_PERTHREAD: "2"
NCCL_BUFFSIZE: "8388608"     # 8MB 버퍼
NCCL_CROSS_NIC: "1"          # 다중 NIC 활용

# Ray 안정성
RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE: "1"
RAY_ENABLE_RECORD_TASK_EVENTS: "1"
```

## 📈 성능 최적화 전략

### **Replica 수 결정 가이드**
```bash
# GPU 가용성 확인
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu"

# 권장 공식
Available_GPUs = Total_GPUs × 0.8  # 20% 여유
Optimal_Replicas = Available_GPUs ÷ (TP × PP)
Max_Replicas = Available_GPUs ÷ (TP × PP)

# 예시: 160 GPU 환경
Optimal_Replicas = 128 ÷ 32 = 4
Max_Replicas = 160 ÷ 32 = 5
```

### **처리량 vs 지연시간 최적화**
```yaml
# 처리량 우선 (배치 처리)
MAX_NUM_SEQS: 128
GPU_MEMORY_UTILIZATION: 0.95

# 지연시간 우선 (실시간 서빙)
MAX_NUM_SEQS: 32
GPU_MEMORY_UTILIZATION: 0.85
```

## 🔧 운영 가이드

### **initContainer 제거 (중요!)**
```yaml
# ❌ 문제: initContainer에서 IB 체크 실패
initContainers:
- name: check-ib
  # BackOff 오류 발생 가능

# ✅ 해결: main container에서 체크
containers:
- name: main
  command: ["/bin/bash", "-lc"]
  args:
  - |
    # IB 체크 후 TCP fallback 설정
    if [ ! -d /sys/class/infiniband ]; then
      export NCCL_IB_DISABLE=1
    fi
    # 정상 시작
```

### **배포 순서**
1. **Head 배포**: `kubectl apply -f head_statefulset.yaml`
2. **Worker 배포**: `kubectl apply -f worker_deployment.yaml`
3. **GPU 확인**: 최소 96 GPU 가용성 체크
4. **Serve 배포**: `kubectl apply -f serve_deepseek.yaml`

### **상태 모니터링**
```bash
# GPU 사용량 확인
kubectl exec -it ray-ku-junyoung-head-0 -n p-ncai-wbl -- ray status

# Serve 상태 확인
kubectl exec -it deployment/ray-deepseek-serve -n p-ncai-wbl -- ray serve status

# Pod 안정성 모니터링
kubectl get pods -n p-ncai-wbl -l ray-cluster=ray-ku-junyoung -w
```

### **문제 해결**
```bash
# OOM 발생시
# 1. GPU_MEMORY_UTILIZATION 감소 (0.90 → 0.85)
# 2. MAX_NUM_SEQS 감소 (64 → 32)
# 3. 메모리 리소스 증가

# 통신 오류시
# 1. IB/RDMA 상태 확인
# 2. NCCL 디버그 로그 확인
# 3. 네트워크 연결성 테스트

# 느린 시작시
# 1. startupProbe failureThreshold 증가
# 2. 모델 캐시 확인
# 3. 디스크 I/O 성능 확인
```

## 🎯 성능 벤치마크

### **예상 성능 (TP=16, PP=2, Replicas=3)**
- **처리량**: ~300 tokens/sec per replica
- **총 처리량**: ~900 tokens/sec
- **지연시간**: ~2-3초 (16K context)
- **동시 요청**: ~192개 (64 × 3)

### **스케일링 가이드**
```yaml
# 처리량 2배 증가 필요시
NUM_REPLICAS: 6  # 192 GPU 필요

# 지연시간 50% 감소 필요시
PIPELINE_PARALLEL_SIZE: 1  # TP=32, PP=1
TENSOR_PARALLEL_SIZE: 32
```

## ✅ 체크리스트

- [ ] GPU 가용성: 최소 96개 확인
- [ ] 메모리: 노드당 최소 512GB
- [ ] 네트워크: IB/RDMA 활성화 확인
- [ ] 스토리지: 모델 캐시용 고속 디스크
- [ ] 모니터링: Prometheus/Grafana 설정
- [ ] 백업: 모델 가중치 백업 계획
