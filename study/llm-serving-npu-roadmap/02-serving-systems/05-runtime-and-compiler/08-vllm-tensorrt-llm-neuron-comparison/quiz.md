# vLLM, TensorRT-LLM, Neuron 비교 퀴즈

## 객관식 1
Neuron 2.31의 현재 serving 기준선으로 가장 정확한 것은 무엇인가?

1. 지원이 종료된 vLLM V0 container
2. vLLM V1 기반의 독립 `vllm-neuron` plugin
3. TensorRT-LLM engine을 Trainium에서 그대로 실행하는 경로
4. compiler 없이 eager mode만 사용하는 NxD Training

## 객관식 2
TensorRT-LLM의 disaggregated serving을 검토할 때 반드시 새 비용으로 측정해야 하는 것은 무엇인가?

1. tokenizer vocabulary 크기만
2. KV state transfer와 stage 사이 handoff
3. 모델 다운로드 시간만
4. prompt의 언어만

## 객관식 3
Neuron 배포 버전 계약으로 가장 적절한 것은 무엇인가?

1. model 이름만 기록한다.
2. upstream vLLM 버전만 기록한다.
3. Neuron SDK, plugin, vLLM, hardware generation, compiled artifact를 함께 기록한다.
4. latest tag를 매번 자동 설치하고 artifact를 무조건 재사용한다.

## 짧은 서술형 1
공통 `vllm serve` API가 GPU와 Trainium의 운영 절차까지 같게 만들지 못하는 이유를 설명하라.

## 짧은 서술형 2
Neuron 2.28의 V0 지원 종료가 기존 deployment에 미치는 영향을 migration checklist 형태로 적어라.

## 심화 설명형
모델 교체가 잦은 현재 vLLM 서비스가 장기적으로 TensorRT-LLM 또는 vLLM Neuron으로 이동하려 한다. production trace를 사용한 비교 실험을 설계하고 model support, feature parity, p99, artifact, migration 비용을 어떻게 판정할지 설명하라.

## 정답 및 해설
- 객관식 1 정답: 2. Neuron 2.31은 vLLM V1 기반 `vllm-neuron` Beta plugin을 도입했으며 NxD Inference에서의 migration 경로를 별도로 제공한다. [S4][S6]
- 객관식 2 정답: 2. prefill과 decode를 분리하면 KV transfer와 router/handoff가 새 latency 및 장애 경계가 된다. [S2]
- 객관식 3 정답: 3. compiler backend는 package 하나가 아니라 SDK, plugin, vLLM, target hardware와 artifact 조합으로 재현된다. [S3][S6]
- 짧은 서술형 1 예시: API는 client request 형식을 통일하지만 backend의 compile 방식과 communication을 통일하지 않는다. Neuron은 bucket별 NEFF와 SDK/plugin matrix가 필요하고, TensorRT-LLM은 engine build와 KV handoff topology를 별도로 관리한다. [S2][S3]
- 짧은 서술형 2 예시: V0 container와 fork 사용 여부를 inventory하고 V1 plugin의 model-feature compatibility를 확인한다. 새 environment에서 correctness, latency, cache와 speculative decoding을 다시 검증하고 artifact 및 rollback 계획을 기록한다. [S4][S5]
- 심화 설명형 해설 포인트: 같은 model revision과 traffic trace를 고정하고 vLLM 기준선을 만든다. 각 target의 지원 feature, build/compile artifact, p50/p95/p99, quality와 장애 복구를 측정하며, version pin과 운영자 migration 시간을 비용에 포함한다.
