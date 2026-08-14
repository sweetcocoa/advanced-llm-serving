# 서빙 시스템 설계

이 트랙은 학습이 끝난 모델을 실제 요청 경로에서 빠르고 안정적으로 실행하는 방법을 다룬다. cache, batching, scheduling, parallelism, GPU/NPU, runtime, compiler와 운영 관측을 하나의 serving system으로 연결한다.

## 이 트랙을 마치면 설명할 수 있어야 하는 것
- 모델별 inference workload를 compute, memory, mutable state 관점으로 분해할 수 있다.
- cache와 scheduler가 latency, throughput, goodput을 어떻게 바꾸는지 설명할 수 있다.
- GPU와 NPU를 operator support, memory hierarchy, lowering contract로 비교할 수 있다.
- runtime과 compiler, observability를 포함한 production serving 경로를 설계할 수 있다.

## 섹션 순서
- [Inference Foundations](./01-inference-foundations/README.md)
- [Cache, Batching, and Generation](./02-cache-batching-and-generation/README.md)
- [Distributed Serving](./03-distributed-serving/README.md)
- [Optimization and Accelerators](./04-optimization-and-accelerators/README.md)
- [Runtime and Compiler](./05-runtime-and-compiler/README.md)
- [Production Operations](./06-production-operations/README.md)

## NPU의 위치
NPU는 별도 모델 계보가 아니다. 모델 graph를 어떤 정밀도와 operator 경계로 lowering하고 어느 runtime에 배치할지 결정하는 serving accelerator이므로 이 트랙 안에 둔다.
