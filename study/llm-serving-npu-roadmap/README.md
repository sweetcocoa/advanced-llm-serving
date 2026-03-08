# LLM Serving + NPU 대화형 수업 로드맵

이 문서 세트는 2026년 3월 8일 기준으로 `LLM Serving`과 `NPU`를 하나의 추론 시스템 관점에서 학습하도록 구성한 대화형 수업 노트다.

## 읽는 방법
- 각 챕터는 `교수자`와 `학습자`가 실제 수업처럼 대화하는 형식으로 작성됐다.
- 모든 챕터는 핵심 수식 2개 이상, Mermaid 다이어그램 2개 이상, 외부 참고 이미지 2개 이상을 포함한다.
- 본문 기술 주장에는 `[S1]`, `[S2]` 형식의 텍스트 출처를 붙이고, 이미지 출처는 각 챕터 `assets/sources.md`에 따로 둔다.

## 권장 학습 순서
- [01-foundations](./01-foundations/README.md): 기초와 성능 직관
- [02-llm-serving-core](./02-llm-serving-core/README.md): LLM Serving 핵심
- [03-llm-serving-advanced](./03-llm-serving-advanced/README.md): 고급 Serving 패턴
- [04-serving-systems](./04-serving-systems/README.md): Serving 시스템 설계
- [05-npu-stack](./05-npu-stack/README.md): NPU 소프트웨어 스택
- [06-runtime-and-compiler](./06-runtime-and-compiler/README.md): 런타임과 컴파일러
- [07-synthesis](./07-synthesis/README.md): 종합 정리와 최신 흐름

## 완주 기준
- prefill/decode, KV cache, batching, quantization, runtime partition을 한 흐름으로 설명할 수 있다.
- cloud serving과 edge/NPU 배포를 같은 추론 파이프라인의 다른 배치로 비교할 수 있다.
- 2026년 최신 기능이 어떤 병목을 겨냥하는지 공식 문서 기준으로 설명할 수 있다.
