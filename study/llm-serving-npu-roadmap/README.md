# LLM Serving + NPU 대화형 수업 로드맵

이 문서 세트는 2026년 8월 12일 기준으로 `LLM Serving`과 `NPU`를 하나의 추론 시스템 관점에서 학습하도록 구성한 대화형 수업 노트다.

## 읽는 방법
- 각 챕터는 `교수자`와 `학습자`가 실제 수업처럼 대화하는 형식으로 작성됐다.
- 대부분의 본문 챕터는 핵심 수식과 Mermaid 다이어그램을 포함한다. 외부 이미지는 설명 가치가 있을 때만 사용하고, 단순 로고나 장식 이미지는 늘리지 않는다.
- 본문 기술 주장에는 `[S1]`, `[S2]` 형식의 텍스트 출처를 붙이고, 이미지 출처는 각 챕터 `assets/sources.md`에 따로 둔다.
- 2026년 상반기 이후 추가되는 내용은 공식 문서, 벤더 기술 블로그, 연구 논문을 우선 근거로 삼고, 추론이나 운영 휴리스틱은 본문에서 명시적으로 구분한다.

## 권장 학습 순서
- [01-foundations](./01-foundations/README.md): 기초와 성능 직관
- [02-llm-serving-core](./02-llm-serving-core/README.md): LLM Serving 핵심
- [03-llm-serving-advanced](./03-llm-serving-advanced/README.md): 고급 Serving 패턴
- [04-serving-systems](./04-serving-systems/README.md): Serving 시스템 설계
- [05-npu-stack](./05-npu-stack/README.md): NPU 소프트웨어 스택
- [06-runtime-and-compiler](./06-runtime-and-compiler/README.md): 런타임과 컴파일러
- [07-synthesis](./07-synthesis/README.md): 종합 정리와 최신 흐름
- [08-iterative-generation-models](./08-iterative-generation-models/README.md): Diffusion LLM과 recurrent-depth Transformer라는 새로운 LLM 계보

## 완주 기준
- prefill/decode, KV cache, batching, quantization, runtime partition을 한 흐름으로 설명할 수 있다.
- cloud serving과 edge/NPU 배포를 같은 추론 파이프라인의 다른 배치로 비교할 수 있다.
- 2026년 최신 기능이 어떤 병목을 겨냥하는지 공식 문서 기준으로 설명할 수 있다.
- autoregressive, diffusion, recurrent-depth LLM을 서로 다른 생성·계산 계보로 설명하고 현재 serving 지원 수준을 비교할 수 있다.
- diffusion LLM과 looped Transformer가 상태를 반복 갱신한다는 점에서는 닮았지만, 상태 공간과 학습 목표, causality는 다르다는 것을 설명할 수 있다.
