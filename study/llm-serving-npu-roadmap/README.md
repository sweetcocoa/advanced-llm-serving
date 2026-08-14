# LLM 모델과 서빙 시스템 대화형 수업 로드맵

이 문서 세트는 2026년 8월 14일 기준으로 LLM이 무엇을 학습하고 생성하는지 설명하는 `모델 이론`과, 학습된 모델을 실제 하드웨어에서 실행하는 `서빙 시스템 이론`을 분리해 학습하도록 구성한 대화형 수업 노트다. 마지막 트랙에서는 두 영역을 제품 설계 문제로 다시 결합한다.

## 읽는 방법
- 각 챕터는 `교수자`와 `학습자`가 실제 수업처럼 대화하는 형식으로 작성됐다.
- 대부분의 본문 챕터는 핵심 수식과 Mermaid 다이어그램을 포함한다. 외부 이미지는 설명 가치가 있을 때만 사용하고, 단순 로고나 장식 이미지는 늘리지 않는다.
- 본문 기술 주장에는 `[S1]`, `[S2]` 형식의 텍스트 출처를 붙이고, 이미지 출처는 각 챕터 `assets/sources.md`에 따로 둔다.
- 2026년 상반기 이후 추가되는 내용은 공식 문서, 벤더 기술 블로그, 연구 논문을 우선 근거로 삼고, 추론이나 운영 휴리스틱은 본문에서 명시적으로 구분한다.

## 권장 학습 순서
- [01-llm-models](./01-llm-models/README.md): autoregressive Transformer, MoE, Diffusion LLM, recurrent-depth 모델의 architecture와 generation contract
- [02-serving-systems](./02-serving-systems/README.md): cache, batching, scheduling, distributed serving, GPU/NPU, runtime, compiler와 production operations
- [03-system-design](./03-system-design/README.md): cloud/edge 배치, 제품 시나리오, frontier 평가와 도입 판단

## 두 이론의 경계
- 학습 objective, 확률 분해, attention mask, router, hidden state와 종료 조건이 바뀌면 `LLM 모델 이론`에서 다룬다.
- 같은 모델을 cache, batching, parallelism, quantization, accelerator와 runtime으로 실행하는 문제는 `서빙 시스템 설계`에서 다룬다.
- MoE, long context, Diffusion LLM처럼 두 영역이 만나는 주제는 한 챕터에 섞지 않고 모델 contract와 runtime 구현을 별도 챕터로 연결한다.

## 완주 기준
- prefill/decode, KV cache, batching, quantization, runtime partition을 한 흐름으로 설명할 수 있다.
- cloud serving과 edge/NPU 배포를 같은 추론 파이프라인의 다른 배치로 비교할 수 있다.
- 2026년 최신 기능이 어떤 병목을 겨냥하는지 공식 문서 기준으로 설명할 수 있다.
- autoregressive, diffusion, recurrent-depth LLM을 서로 다른 생성·계산 계보로 설명하고 현재 serving 지원 수준을 비교할 수 있다.
- diffusion LLM과 looped Transformer가 상태를 반복 갱신한다는 점에서는 닮았지만, 상태 공간과 학습 목표, causality는 다르다는 것을 설명할 수 있다.
- NPU가 별도 모델 계보가 아니라 graph lowering과 execution placement를 담당하는 serving accelerator인 이유를 설명할 수 있다.
