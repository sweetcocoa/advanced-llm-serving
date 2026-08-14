# Disaggregated Prefill/Decode Quiz

## 객관식 1
다음 중 disaggregated prefill/decode를 도입하는 가장 직접적인 이유로 가장 알맞은 것은 무엇인가?

A. 모델 파라미터 수를 줄여 정확도를 높이기 위해  
B. prefill과 decode가 만드는 서로 다른 대기열과 자원 병목을 분리하기 위해  
C. speculative decoding의 승인율을 자동으로 높이기 위해  
D. structured output의 schema 검증을 제거하기 위해

## 객관식 2
다음 workload 중 disaggregated prefill/decode의 ROI가 가장 높을 가능성이 큰 것은 무엇인가?

A. 입력 800토큰, 출력 20토큰의 짧은 JSON 응답  
B. 입력 1k 토큰, 출력 900토큰의 자유 서술 응답  
C. 입력 50k 토큰, 출력 120토큰의 계약서 위험 조항 요약  
D. 입력 300토큰, 출력 30토큰의 tool calling 응답

## 객관식 3
다음 중 prefix caching 또는 KV cache reuse와 disaggregated prefill/decode의 관계를 가장 정확히 설명한 것은 무엇인가?

A. 둘은 완전히 같은 기법이라 이름만 다르다.  
B. 둘은 항상 서로를 대체하므로 동시에 쓸 필요가 없다.  
C. prefix/KV reuse는 재계산을 줄이고, disaggregation은 자원 간섭을 줄이므로 함께 쓰일 수 있다.  
D. disaggregation은 KV cache를 사용하지 않는 환경에서만 가능하다.

## 짧은 서술형 1
`shared queue 절감 > KV 이동 비용`이라는 조건이 왜 disaggregated prefill/decode의 핵심 판단 기준인지 3~4문장으로 설명하시오.

## 짧은 서술형 2
멀티테넌트 환경에서 긴 RAG 요청과 짧은 채팅 요청이 같은 GPU 풀을 공유할 때 어떤 tail latency 문제가 생길 수 있는지 설명하시오.

## 심화 설명형 1
다음 두 서비스를 비교해, 어느 쪽에서 disaggregated prefill/decode가 더 우선순위가 높은지 설명하시오.

- 서비스 A: 입력 40k 토큰, 출력 100토큰의 문서 요약
- 서비스 B: 입력 1k 토큰, 출력 700토큰의 장문 assistant 응답

답변에는 최소한 다음을 포함하시오.
- prefill/decode 중 어느 단계가 더 지배적인지
- speculative decoding과의 우선순위 비교
- KV 이동 비용을 감수할 가치가 있는지 판단 근거

## 심화 설명형 2
structured outputs와 tool calling이 많은 agent 시스템에서 disaggregated prefill/decode를 검토할 때, 어떤 route는 좋은 후보가 되고 어떤 route는 아닐 수 있다. 이 차이를 입력 길이, 출력 길이, 제약된 출력 형식 관점에서 설명하시오.

## 정답 및 해설
### 객관식 정답
- 객관식 1: B
- 객관식 2: C
- 객관식 3: C

### 해설
**객관식 1 해설:** disaggregated prefill/decode의 본질은 모델 정확도 개선이 아니라, prefill과 decode가 서로 다른 큐와 자원 압력을 만든다는 점을 운영 설계에 반영하는 것이다. 긴 입력이 shared GPU를 오래 붙잡는 상황에서 decode 세션과의 간섭을 줄이는 것이 핵심이다 [S1][S2].

**객관식 2 해설:** 입력 50k, 출력 120처럼 `긴 입력 + 짧은 출력` 조합은 prefill 병목이 강하고, shared queue를 오래 점유하기 쉽다. 반면 입력 1k, 출력 900은 decode 최적화의 우선순위가 더 높아 speculative decoding을 먼저 볼 가능성이 크다 [S2][S4].

**객관식 3 해설:** prefix caching과 KV cache reuse는 이미 계산한 것을 다시 쓰는 전략이고, disaggregated prefill/decode는 prefill과 decode가 같은 줄에서 기다리지 않게 하는 전략이다. 따라서 두 접근은 서로 다른 축의 최적화이며 함께 설계될 수 있다 [S1][S3].

**짧은 서술형 1 예시 답안:** disaggregation은 shared 구조에서 줄어드는 대기열 시간이 KV를 옮기는 새 비용보다 클 때만 이득이 난다. 즉 prefill과 decode를 분리한다고 해서 자동으로 빨라지는 것이 아니다. 요청 분포가 짧고 균일하면 shared queue 자체가 길지 않아 이득이 작다. 반대로 긴 입력이 섞여 shared queue가 자주 막히면, KV 이동 비용을 내더라도 분리 설계가 더 빠를 수 있다 [S1][S2].

**짧은 서술형 2 예시 답안:** 긴 RAG 요청은 prefill 단계에서 많은 시간을 쓰며 shared GPU 풀의 앞단을 오래 점유한다. 그 결과 이미 응답 중이던 짧은 채팅 세션도 scheduler 경쟁을 겪고 tail latency가 커진다. 사용자 입장에서는 "짧은 질문인데도 응답 시작이 늦다"는 현상으로 보인다. 분리 설계는 이 간섭을 줄이기 위해 prefill 큐와 decode 큐를 나누는 접근이다 [S1][S2].

**심화 설명형 1 예시 답안:** 서비스 A가 disaggregated prefill/decode의 우선순위가 더 높다. 입력 40k, 출력 100이면 prefill이 latency 대부분을 차지할 가능성이 높고, 긴 입력이 shared queue를 막는 효과도 크다. 따라서 KV 이동 비용이 추가되더라도, prefill 전용 풀과 decode 전용 풀로 나누어 다른 요청과의 간섭을 줄일 가치가 크다. 반면 서비스 B는 입력이 짧고 출력이 길어 decode 비중이 커지므로, 먼저 speculative decoding 같은 decode 최적화를 고려하는 편이 자연스럽다. 이 경우 disaggregation보다 target의 순차 decode 횟수를 줄이는 쪽이 더 직접적인 이득을 줄 수 있다 [S2][S4].

**심화 설명형 2 예시 답안:** agent 시스템에서도 긴 문서를 읽고 마지막에 짧은 함수 호출 JSON을 내보내는 route는 prefill 중심 병목을 만들 수 있어 disaggregation 후보가 된다. 반대로 입력도 짧고 출력도 매우 짧은 structured response route는 shared queue 절감이 작아서 KV 이동 비용이 더 두드러질 수 있다. structured outputs와 tool calling은 출력 형식을 제약하므로 decode 길이가 길지 않은 경우가 많고, 이때는 분리 설계보다 route 분리나 캐시 재사용, 혹은 단순한 스케줄 조정이 더 나을 수 있다. 결국 agent route도 `얼마나 많이 읽는가`와 `얼마나 길게 생성하는가`를 분리해서 판단해야 한다 [S5][S6].
