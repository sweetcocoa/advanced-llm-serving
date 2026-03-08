# Long Context and Memory Pressure 퀴즈

## 객관식 1
다음 중 `긴 사내 규정 QA` 서비스에서 가장 먼저 점검할 대상으로 가장 적절한 것은 무엇인가?

1. decode context parallel의 통신량
2. block 단위 common prefix와 KV reuse 가능 시점
3. tool result retention 정책
4. speculative decoding 승인율

## 객관식 2
다음 중 `host offload`에 대한 설명으로 가장 적절한 것은 무엇인가?

1. KV를 버리고 필요하면 prompt 전체를 다시 prefill한다.
2. prefill worker를 decode worker와 분리하는 구조와 같은 말이다.
3. reusable KV의 상주 위치를 GPU 밖으로 옮겨 GPU pressure를 낮추는 방식이다.
4. shared prefix hit가 낮을수록 무조건 유리해진다.

## 객관식 3
prefix 재사용이 거의 없는 장문 요약 서비스에서 TTFT가 높다. 다음 중 가장 적절한 진단 순서는 무엇인가?

1. 먼저 disaggregated serving을 넣고, 효과가 없으면 원인을 다시 찾는다.
2. prompt 계산이 병목인지, prefill/decode 자원 간섭이 병목인지 나눈다.
3. decode context parallel부터 넣어 prefill 시간을 줄인다.
4. host offload만 적용하면 TTFT 문제도 함께 해결된다.

## 짧은 서술형 1
`긴 상담 세션` 서비스에서 왜 `decode KV capacity pressure`를 따로 진단해야 하는지 2~3문장으로 설명하라. 답변에는 `live KV`, `GPU`, `decode`를 포함하라.

## 짧은 서술형 2
`shared prefix가 긴 서비스`에서 "문자열이 비슷하다"보다 "block 단위 공통 prefix"를 봐야 하는 이유를 설명하라. 답변에는 `reuse`, `block`, `prefill`을 포함하라.

## 심화 설명형 1
다음 상황에 대해 어떤 선택지를 먼저 비교할지 설명하라.

- 상황 A: 48K 정책 문서가 매번 앞부분에 붙고, 답변은 짧다.
- 상황 B: 대화 길이가 길어질수록 GPU 메모리 부족으로 동시 세션 수가 급격히 줄어든다.

답변에는 `KV reuse`, `host offload`, `decode context parallel`, `disaggregated serving` 중 최소 세 가지를 포함하라.

## 심화 설명형 2
어떤 팀이 "긴 context라서 느리다"는 이유만으로 disaggregated serving을 도입했지만 개선이 작았다. 이 챕터의 관점에서 가능한 원인을 설명하라.

답변에는 `prefill compute`, `KV handoff`, `shared prefix`, `phase interference` 중 최소 세 가지를 포함하라.

## 정답 및 해설
- 객관식 1 정답: 2. shared prefix가 긴 서비스의 첫 질문은 문자열 유사성보다 block 단위 공통 prefix를 얼마나 재사용할 수 있는가이다. TensorRT-LLM의 KV reuse와 KV cache system, vLLM의 KV Cache Manager가 이 관점을 뒷받침한다. [S3][S4][S6]
- 객관식 2 정답: 3. host offload는 KV를 폐기하는 기능이 아니라 GPU 밖으로 내려 두었다가 필요 시 다시 접근하는 흐름이다. 따라서 줄이는 것은 GPU resident pressure이고, 대가로 복사 비용이 생긴다. [S4]
- 객관식 3 정답: 2. prefix reuse가 약한 장문 요약이라면 먼저 prompt 계산 자체가 무거운지, 아니면 prefill/decode가 같은 자원 풀에서 부딪히는지 나눠야 한다. 전자는 prefill context parallel 쪽이고, 후자는 disaggregated serving 쪽이다. [S1][S2][S5]
- 짧은 서술형 1 예시: 긴 상담 세션에서는 decode가 오래 이어지면서 live KV가 GPU에 계속 남는다. 그래서 문제는 "prefill을 빨리 끝내는가"보다 "decode 동안 유지되는 KV를 GPU에 얼마나 오래 둘 것인가"가 된다. 이 경우 decode KV capacity pressure를 따로 봐야 offload나 decode context parallel 같은 선택지가 보인다. [S4][S5]
- 짧은 서술형 2 예시: 긴 서비스에서는 prefix의 앞부분이 block 경계까지 얼마나 유지되는지가 reuse 가능성을 좌우한다. 문자열이 비슷해 보여도 block 단위 공통 prefix가 맞지 않으면 prefill 재계산을 충분히 줄이지 못할 수 있다. [S3][S4][S6]
- 심화 설명형 1 예시: 상황 A는 KV reuse를 먼저 본다. 긴 정책 문서의 앞부분이 반복되므로 block 기반 reuse가 prefill 부담을 가장 직접적으로 낮출 수 있다. GPU pressure가 함께 보이면 host offload를 두 번째 카드로 비교한다. 상황 B는 decode context parallel과 host offload가 먼저다. 여기서는 긴 세션의 live KV가 GPU를 잠그는 것이 핵심이므로, disaggregated serving은 보조 후보일 뿐 중심 답은 아니다. [S1][S3][S4][S5][S6]
- 심화 설명형 2 예시: 병목이 prefill compute인데 disaggregation을 넣으면, 계산 자체는 그대로 두고 KV handoff 경로만 추가할 수 있다. 또 shared prefix가 긴 서비스였다면 reuse를 먼저 개선하는 편이 더 직접적일 수 있다. disaggregated serving은 phase interference를 줄일 때 강하지만, 그 항이 실제 병목이 아니면 효과가 제한된다. [S1][S2][S3][S5]
