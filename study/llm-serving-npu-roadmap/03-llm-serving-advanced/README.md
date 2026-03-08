# 고급 Serving 패턴

reuse, speculative execution, disaggregation, MoE와 같이 2026년 실무에서 중요한 고급 serving 패턴을 다룬다.

## 이 모듈을 마치면 설명할 수 있어야 하는 것
- prefix caching과 KV cache reuse 차이를 설명할 수 있다.
- speculative decoding이 왜 latency를 줄이는지 수식으로 설명할 수 있다.
- MoE, long context, structured output이 serving에 추가하는 제약을 설명할 수 있다.

## 챕터 순서
- [Prefix Caching](./01-prefix-caching/README.md): 같은 system prompt와 context를 재사용하는 prefix cache 전략을 본다.
- [Speculative Decoding](./02-speculative-decoding/README.md): draft model과 target model의 협업으로 decode latency를 줄이는 원리를 본다.
- [Disaggregated Prefill/Decode](./03-disaggregated-prefill-decode/README.md): prefill과 decode를 다른 자원에 분리하는 시스템 설계를 정리한다.
- [MoE and Expert Parallelism](./04-moe-and-expert-parallelism/README.md): MoE 모델의 활성 파라미터와 expert routing을 serving 관점에서 본다.
- [Structured Outputs and Tool Calling](./05-structured-outputs-and-tool-calling/README.md): structured output과 tool calling이 decoding 제약과 latency에 미치는 영향을 본다.
- [Long Context and Memory Pressure](./06-long-context-and-memory-pressure/README.md): 긴 입력이 prefill 자원 사용과 KV 이동 비용을 어떻게 키우는지 다룬다.

## 선행 관계
- 같은 모듈에서는 위 순서를 따르는 것이 좋다.
- 다음 모듈로 넘어가기 전에 각 챕터 퀴즈를 먼저 풀어 본다.
