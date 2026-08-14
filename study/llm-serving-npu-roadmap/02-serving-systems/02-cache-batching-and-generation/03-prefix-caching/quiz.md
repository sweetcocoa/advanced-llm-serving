## 객관식 1
고객지원 assistant가 매 요청마다 같은 system prompt를 사용한다고 생각했는데, 실제 prefix cache hit율이 낮았다. 가장 먼저 의심할 원인으로 가장 적절한 것은 무엇인가?

1. decode 단계에서 생성 길이가 짧다
2. system prompt 내부에 현재 시각, 사용자 ID, 실험군 플래그 같은 동적 값이 들어간다
3. GPU 수가 충분하지 않다
4. speculative decoding을 사용하지 않는다

## 객관식 2
다음 중 prefix caching과 KV cache reuse의 관계를 가장 잘 설명한 것은 무엇인가?

1. 두 용어는 완전히 같은 뜻이라 운영 관점에서 구분할 필요가 없다
2. prefix caching은 decode 최적화이고, KV cache reuse는 prefill 최적화다
3. prefix caching은 공통 앞부분의 재사용에 초점을 두고, KV cache reuse는 더 넓은 KV 상태 재활용 관점으로 볼 수 있다
4. prefix caching은 tool calling에서만 쓰이고, KV cache reuse는 일반 챗봇에서만 쓰인다

## 객관식 3
어떤 서비스의 평균 시간은 `T_prefill = 120ms`, `T_decode = 80ms`, prefix cache hit율은 `h = 0.75`다. 이때 단순 모델 `E[T] = T_decode + (1-h)T_prefill`로 계산한 기대 latency는 얼마인가?

1. 90ms
2. 110ms
3. 170ms
4. 200ms

## 짧은 서술형 1
structured output 또는 tool calling이 prefix cache hit율을 떨어뜨릴 수 있는 이유를 2~3문장으로 설명하라.

## 짧은 서술형 2
prefix cache miss가 아직 많은 서비스에서도 disaggregated prefill/decode를 함께 검토할 가치가 있는 이유를 2~3문장으로 설명하라.

## 심화 설명형 1
사내 정책 Q&A 봇이 있다. 공통 system prompt와 규정집 요약을 붙인 뒤 사용자 질문을 받는다. 최근 hit율이 급감했고, 응답 지연도 늘었다. 아래 조건을 반영해 디버깅 순서를 설계하라.

- 지난 배포에서 규정집 버전 표기 방식이 바뀌었다.
- structured output 스키마도 조금 수정되었다.
- prefill worker queue 길이가 함께 증가했다.

## 심화 설명형 2
다음 두 워크로드 중 어느 쪽이 prefix caching에서 더 큰 효과를 보기 쉬운지 비교하고, 이유를 설명하라.

- 워크로드 A: 매 요청마다 긴 원문 문서 전체가 달라지는 요약 서비스
- 워크로드 B: 같은 tool 설명, 같은 응답 정책, 같은 사내 정책 문서를 붙이고 마지막 사용자 질문만 달라지는 assistant

또한 효과가 큰 쪽에서도 어떤 이유로 hit율이 깨질 수 있는지 한 가지 이상 덧붙여라.

## 정답 및 해설
### 객관식 정답
- 객관식 1: 2번
- 객관식 2: 3번
- 객관식 3: 2번

### 객관식 해설
- 객관식 1: prefix caching은 반복되는 앞부분이 핵심이다. 따라서 dynamic field가 prefix 해시를 흔드는지 먼저 봐야 한다. 날짜, 사용자별 메타데이터, 실험군 플래그는 대표적인 invalidation 원인이다.
- 객관식 2: 이 챕터에서는 prefix caching을 "공통 prefix 표준화" 관점으로, KV cache reuse를 "계산된 KV 상태 재활용" 관점으로 구분했다. 둘은 겹치지만 운영 질문이 다르다. [S3]
- 객관식 3: `E[T] = 80 + (1-0.75) * 120 = 80 + 30 = 110ms`다. 이 식은 prefix caching이 주로 prefill 비용을 줄인다는 점을 보여 준다.

### 짧은 서술형 예시 답안
- 짧은 서술형 1: structured output의 스키마나 tool calling의 함수 목록과 설명이 공통 prefix 안에 들어가면, 배포마다 prefix 내용이 바뀔 수 있다. 그러면 같은 사용자 질문 패턴이 반복돼도 캐시 키가 달라져 hit율이 떨어진다. [S5][S6]
- 짧은 서술형 2: hit율이 낮아도 miss가 몰리는 구간의 병목이 prefill이면 지연과 자원 점유가 커진다. 이때 prefill/decode를 분리하면 miss 비용을 별도 자원에서 흡수할 수 있어 전체 서비스 안정성이 좋아질 수 있다. [S1][S2]

### 심화 설명형 예시 답안
- 심화 설명형 1: 첫 단계는 규정집 버전 표기가 공통 prefix 안에 들어가 해시를 깨는지 확인하는 것이다. 그다음 structured output 스키마 변경이 prefix 일부를 바꿨는지 본다. 이 두 변화가 hit율 저하의 직접 원인인지 확인한 뒤, prefill worker queue 증가와 함께 관측되었다면 miss 증가가 실제 prefill 병목으로 이어졌다고 판단할 수 있다. 마지막으로 동적 값을 prefix 밖으로 빼고, 스키마 버전을 안정적으로 관리하며, 여전히 miss가 많다면 prefill/decode 분리 강화를 검토한다. [S1][S2][S5]
- 심화 설명형 2: 더 큰 효과를 보기 쉬운 쪽은 워크로드 B다. 같은 tool 설명, 응답 정책, 사내 정책 문서처럼 반복되는 공통 앞부분이 크기 때문이다. 반대로 워크로드 A는 길어도 매번 원문이 달라 prefix reuse 여지가 작다. 다만 워크로드 B도 tool 목록 변경, 정책 버전 문구 추가, 사용자별 동적 메타데이터 삽입 같은 이유로 hit율이 쉽게 깨질 수 있다. [S5][S6]
