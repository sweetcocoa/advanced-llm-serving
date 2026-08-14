# Mixture of Experts

MoE를 분산 배치 기법이 아니라 conditional computation 모델로 먼저 읽는다. router, top-k, capacity, load-balancing objective가 정의하는 모델 동작과 expert-parallel serving을 분리한다.

## 챕터 순서
- [MoE Architecture and Conditional Computation](./01-moe-architecture-and-conditional-computation/README.md): sparse gating과 expert 선택이 모델 용량과 계산량을 어떻게 분리하는지 본다.

## 경계
All-to-all, expert placement, EPLB와 tail latency는 `02-serving-systems/03-distributed-serving`에서 다룬다.
