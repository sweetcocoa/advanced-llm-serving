# 서버급 NPU 서빙 퀴즈

## 객관식 1
on-device NPU와 server NPU를 가장 잘 구분한 설명은 무엇인가?

1. server NPU도 CPU fallback만 확인하면 된다.
2. server NPU에서는 rank topology, collective, graph artifact와 cluster failure가 운영 계약에 들어간다.
3. on-device NPU에서만 compiler가 필요하다.
4. 둘은 전력 한도 외에는 완전히 같다.

## 객관식 2
vLLM Ascend 설치 문서가 권장하는 version 관리 방식은 무엇인가?

1. vLLM만 최신으로 설치한다.
2. `vllm-ascend`, vLLM, PyTorch, torch-npu, CANN을 검증된 compatibility set으로 선택한다.
3. package resolver가 성공하면 driver는 기록하지 않는다.
4. main branch에서는 임의의 vLLM release를 사용한다.

## 객관식 3
AFD에 대한 설명으로 가장 정확한 것은 무엇인가?

1. prefill과 decode만 분리한다.
2. Attention과 FFN을 별도 rank group으로 나누며 Ascend connector는 stage와 graph mode별 제한이 있다.
3. 모든 모델과 hardware에서 production-ready다.
4. collective를 제거한다.

## 객관식 4
graph capture의 효과를 평가할 때 함께 보고해야 할 항목은 무엇인가?

1. graph on/off만
2. bucket, hit rate, fallback, warm-up과 artifact identity
3. model 이름만
4. 평균 NPU utilization만

## 짧은 서술형 1
EP rank를 늘렸는데 p99가 나빠진 상황에서 collective 관점의 진단 순서를 설명하라.

## 짧은 서술형 2
Neuron과 Ascend 중 하나를 골라 production version matrix의 항목을 작성하라.

## 심화 설명형 1
Ascend MoE 서비스에 experimental AFD plugin을 도입하는 canary 계획을 작성하라. connector, graph mode, natural routing, correctness, collective p99와 rollback을 포함하라.

## 심화 설명형 2
server NPU deployment가 `실행 가능`에서 `production 승인`으로 넘어가기 위한 gate를 설계하라. graph, artifact, rank failure와 observability를 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. server NPU는 model을 여러 rank/node에 배치하므로 collective와 cluster failure가 request critical path에 들어간다.
- 객관식 2 정답: 2. vLLM Ascend는 framework, torch-npu와 CANN을 한 compatibility row로 선택하라고 명시한다. [S2]
- 객관식 3 정답: 2. AFD는 Attention과 FFN을 분리하고 Ascend의 sync/async connector 및 ACL graph 범위가 다르다. 현재 plugin은 experimental이다. [S5][S6]
- 객관식 4 정답: 2. capture 이득은 padding, miss/fallback, compile/warm-up과 artifact 비용을 제외하고 평가해야 한다. [S7]
- 짧은 서술형 1 예시: router의 token 분포와 rank별 dispatch 양을 확인하고 all-to-all/dispatch/combine의 p50/p99와 rank skew를 본다. intra/inter-node 경계를 나누고 link/device error 및 version mismatch를 확인한 뒤 TP/EP topology를 조정한다. [S3]
- 짧은 서술형 2 예시: Ascend는 hardware/firmware, CANN, torch/torch-npu, vLLM/vllm-ascend, model revision/precision, TP/EP topology와 ACL graph mode를 기록한다. [S2][S4]
- 심화 설명형 1 해설 포인트: official matrix의 exact plugin/runtime을 고정하고 지원 connector와 graph mode를 선택한다. forced balancing이 아닌 production natural-routing trace로 EP baseline과 비교하며 quality, collective p99, graph fallback과 worker failure를 측정하고 독립 rollback 단위를 둔다. [S5][S6]
- 심화 설명형 2 해설 포인트: correctness baseline, representative traffic의 p99, graph hit/fallback, artifact provenance, collective trace와 rank failure recovery를 모두 통과해야 한다. service startup에 전체 version matrix를 출력하고 canary에서 rollback을 검증한다.
