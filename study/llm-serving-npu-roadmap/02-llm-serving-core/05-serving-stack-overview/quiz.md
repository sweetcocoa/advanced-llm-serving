# Serving Stack Overview Quiz

## 객관식 1
한 팀이 `/v1/chat/completions` 엔드포인트를 이미 열었다. 그런데 긴 문서 요청이 몰릴 때 첫 토큰 지연이 커진다. 이 챕터의 관점에서 가장 먼저 떠올려야 할 판단은 무엇인가?

1. API가 같으니 내부 runtime도 같다고 보고 모델 품질만 비교한다.
2. 엔드포인트 호환성과 내부 스택 책임은 다르므로 queue와 prefill 계층을 먼저 분해해 본다.
3. 이미지 리사이즈 설정을 먼저 바꾼다.
4. backend-specific 기능은 항상 과하므로 처음부터 배제한다.

## 객관식 2
다음 중 `KV Cache Reuse`가 직접 겨냥하는 상황으로 가장 알맞은 것은 무엇인가?

1. 인증 헤더가 자주 누락되는 상황
2. 동일한 prefix를 반복 계산해 prefill 부담이 커지는 상황
3. HTTP 응답 코드가 일관되지 않은 상황
4. 클라이언트 SDK 버전이 제각각인 상황

## 객관식 3
다음 중 `disaggregated serving`을 검토할 가능성이 가장 높은 팀은 누구인가?

1. 짧은 질의응답 위주의 사내 챗봇을 빠르게 붙여야 하는 제품 팀
2. 모델 카드 문구를 정리하는 문서 팀
3. 긴 입력이 몰릴 때 prefill과 decode 간섭을 줄이고 싶은 인프라 팀
4. API 키 발급 정책만 다루는 보안 팀

## 짧은 서술형 1
이 챕터가 서빙 스택을 "제품 이름 모음"이 아니라 "책임 분할표"로 다루는 이유를 4문장 안팎으로 설명하라. 답변에는 반드시 `API layer`, `runtime`, `observability`를 포함하라.

## 짧은 서술형 2
아래 수식을 바탕으로, TTFT가 나빠졌을 때 왜 `T_queue`와 `T_prefill`을 먼저 나눠 봐야 하는지 설명하라.

$$
T_{\mathrm{TTFT}} = T_{\mathrm{api}} + T_{\mathrm{queue}} + T_{\mathrm{prefill}} + T_{\mathrm{dispatch}}
$$

답변에는 반드시 `queue depth`, `병목`, `오진`을 포함하라.

## 심화 설명형 1
다음 상황에서 어떤 계층부터 어떤 순서로 확인할지 서술하라.

- 계약서 검토 서비스다.
- 공통 시스템 프롬프트가 길다.
- 긴 첨부 문서가 자주 들어온다.
- 사용자는 첫 토큰이 특히 느리다고 말한다.

답변에는 반드시 `KV cache reuse`, `cache hit ratio`, `disaggregated serving`, `prefill`을 포함하라.

## 심화 설명형 2
다음 수식을 이용해, 왜 어떤 조직은 범용 runtime 최적화 엔진에서 만족하고 어떤 조직은 backend-specific 최적화까지 내려가는지 설명하라.

$$
P_{\mathrm{attainable}} \le \min \left(P_{\mathrm{peak}},\ I \cdot B_{\mathrm{mem}}\right)
$$

답변에는 반드시 `memory-bound`, `운영 복잡도`, `사용자 체감`, `디버깅 순서`를 포함하라.

## 정답 및 해설
### 객관식 정답
- 객관식 1: 2번
- 객관식 2: 2번
- 객관식 3: 3번

### 객관식 해설
- 객관식 1 해설: 이 챕터는 API 호환성과 내부 운영 계층을 분리해서 본다. 같은 엔드포인트를 노출해도 queue, prefill, backend 선택은 달라질 수 있으므로 먼저 시간 항목을 분해해야 한다. [S2] [S3] [S4]
- 객관식 2 해설: `KV Cache Reuse`는 반복 prefix를 다시 계산하지 않도록 하려는 기능이다. 따라서 긴 입력과 반복 prefix가 겹치는 서비스에서 prefill 부담을 줄이는 문맥과 연결된다. [S3]
- 객관식 3 해설: `disaggregated serving`은 prefill과 decode의 간섭을 줄이려는 선택과 연결된다. 긴 입력과 큰 운영 규모를 직접 다루는 인프라 팀이 이 선택을 검토할 가능성이 높다. [S4]

### 짧은 서술형 예시 답안
- 짧은 서술형 1 예시: 이 챕터는 서빙 스택을 제품 이름보다 책임 분할표로 본다. API layer는 계약과 라우팅을 맡고, runtime은 batching과 KV cache 같은 공통 운영을 맡는다. observability는 그 둘과 backend 사이에서 어느 계층이 병목인지 가려 주는 역할을 한다. 그래서 같은 API를 제공해도 운영 계약은 달라질 수 있다. [S1] [S2] [S3] [S4]
- 짧은 서술형 2 예시: TTFT는 하나의 숫자처럼 보이지만 수식상 `T_queue`와 `T_prefill`이 따로 들어 있다. queue depth가 올라간 상황과 긴 입력 때문에 prefill이 늘어난 상황은 병목 위치가 다르므로, 둘을 구분하지 않으면 오진하기 쉽다. 예를 들어 queue 문제를 backend 문제로 착각하면 잘못된 최적화에 시간을 쓰게 된다. 그래서 이 챕터는 TTFT를 계층별로 쪼개 본다. [S2] [S3] [S4]

### 심화 설명형 예시 답안
- 심화 설명형 1 예시: 먼저 API 이상보다 runtime 관측부터 본다. 첫 단계는 queue depth를 확인해 대기 자체가 늘었는지 보는 것이고, 그다음 공통 시스템 프롬프트가 길다면 cache hit ratio를 확인해 `KV cache reuse`가 필요한 상황인지 판단한다. 이어서 긴 첨부 문서 때문에 `prefill`이 길어지고 decode를 밀어내는지 확인한다. 그 간섭이 명확하면 `disaggregated serving`을 검토하는 순서가 맞다. [S3] [S4]
- 심화 설명형 2 예시: 수식은 달성 가능한 성능이 연산 상한보다 메모리 대역폭 상한에 먼저 막힐 수 있음을 보여 준다. 이미 memory-bound라면 backend를 더 미세하게 조정해도 사용자 체감이 제한적일 수 있으므로, 어떤 조직은 범용 runtime 안에서 queue와 cache 정책을 다듬는 쪽으로 충분한 성과를 낸다. 반대로 반복 prefix나 prefill/decode 간섭이 observability로 분명히 보이는 조직은 backend-specific 최적화를 감수할 이유가 생긴다. 결국 선택은 디버깅 순서가 정리되어 있는지와 운영 복잡도를 감당할 수 있는지에 달려 있다. [S1] [S3] [S4]
