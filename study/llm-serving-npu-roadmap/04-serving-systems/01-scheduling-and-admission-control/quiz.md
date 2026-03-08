# Scheduling and Admission Control 퀴즈

## 객관식 1
다음 중 이 챕터에서 [S1]로 직접 연결해 말할 수 있는 설명으로 가장 적절한 것은 무엇인가?

1. vLLM 문서는 tenant deficit과 aging 조합을 표준 정책으로 권장한다.
2. vLLM 문서는 continuous batching과 request scheduling이 serving 기준선이라는 점을 보여 준다.
3. vLLM 문서는 모든 admission control이 token budget 식으로 구현되어야 한다고 단정한다.
4. vLLM 문서는 `retry-after` jitter를 반드시 사용해야 한다고 명시한다.

## 객관식 2
다음 중 `실무적 추론`으로 읽어야 할 문장은 무엇인가?

1. prefill/decode를 분리한 비교가 TensorRT-LLM 문서에 있다.
2. continuous batching이 scheduler 기준선이라는 설명이 있다.
3. 동일한 `retry-after` 값은 synchronized retry를 만들어 queue oscillation을 키울 수 있다.
4. disaggregated serving이 phase 분리 비교라는 설명이 있다.

## 객관식 3
멀티테넌트 환경에서 cost-first 정책만 오래 유지했을 때 가장 먼저 걱정해야 할 실패 모양은 무엇인가?

1. 모든 tenant의 완료율이 자동으로 동일해진다.
2. 긴 prompt tenant가 거의 전진하지 못하는 starvation이 나타날 수 있다.
3. prefill과 decode가 자동으로 분리된다.
4. request count와 token backlog가 항상 같은 값이 된다.

## 짧은 서술형 1
왜 queue length만 보면 admission 문제와 scheduler 문제를 섞어 보기 쉬운가? `request count`, `token backlog`, `active decode`라는 단어를 포함해 3~4문장으로 설명하라.

## 짧은 서술형 2
고정 `retry-after`가 왜 단순한 에러 응답이 아니라 다음 부하 파형의 일부가 될 수 있는지 3~4문장으로 설명하라.

## 심화 설명형 1
다음 상황에서 어떤 순서로 원인을 분해할지 설명하라.

- 상황: 평균 latency는 조금만 늘었는데, 특정 tenant의 timeout이 급증했다.
- 조건: 짧은 채팅 요청과 긴 문서 요약 요청이 같은 모델 풀에 섞여 있다.
- 요구: scheduler 문제와 admission 문제를 구분하는 진단 순서를 제시하라.

답변에는 다음이 들어가야 한다.

1. request count와 token backlog를 어떻게 나눠 볼지
2. active decode와 신규 prefill을 어떻게 구분할지
3. starvation 여부를 어떤 관찰로 판단할지
4. 공정성 큐나 aging을 말할 때 왜 `운영 휴리스틱`이라고 불러야 하는지

## 심화 설명형 2
다음 두 선택지를 비교하라.

- 선택지 A: shared pool을 유지한 채 비용 우선 scheduler를 더 강하게 적용한다.
- 선택지 B: prefill/decode를 분리하고 phase별 admission budget을 둔다.

상황은 다음과 같다.

- 긴 분석 요청과 짧은 대화 요청이 같이 들어온다.
- decode 응답성 저하가 특히 민감하다.

답변에는 다음이 들어가야 한다.

1. A의 장점과 위험
2. B의 장점과 새 운영 비용
3. [S2]로 직접 말할 수 있는 부분과, 직접 출처 없이 운영 판단으로 남는 부분의 구분

## 정답 및 해설
- 객관식 1 정답: 2
- 객관식 2 정답: 3
- 객관식 3 정답: 2

- 객관식 1 해설: 이 챕터에서 [S1]은 continuous batching과 request scheduling의 기준선에만 좁게 연결하는 것이 안전하다. tenant deficit, token budget 공식, `retry-after` jitter는 본문에서도 직접 출처가 아니라 추론 또는 휴리스틱으로 분리했다.

- 객관식 2 해설: 동일한 `retry-after` 값이 synchronized retry를 만들 수 있다는 설명은 문서 제목 자체에서 직접 확정되는 사실이 아니라, admission 제어를 time-domain failure로 읽는 `실무적 추론`이다.

- 객관식 3 해설: 비용 우선 정책은 평균 효율에는 도움이 될 수 있지만, 긴 prompt tenant의 진행률을 낮춰 starvation을 만들 수 있다. 이 실패는 "모델이 느려서"가 아니라 "누구를 먼저 서비스할지"라는 정책에서 생긴다.

- 짧은 서술형 1 예시: request count는 큐에 몇 건이 있는지만 보여 주고, 각 요청의 실제 비용 차이는 숨긴다. token backlog는 앞으로 처리해야 할 prompt와 남은 decode 부담을 더 직접적으로 반영한다. active decode가 이미 많이 자원을 붙잡고 있으면 queue length가 짧아 보여도 admission을 더 열면 위험할 수 있다. 그래서 queue length만 보면 scheduler 불공정과 admission 과수용을 쉽게 섞어 보게 된다.

- 짧은 서술형 2 예시: 같은 순간에 거절된 요청이 모두 같은 `retry-after` 값을 받으면, 같은 시각에 다시 몰릴 가능성이 커진다. 그러면 admission은 혼잡을 줄이기보다 주기적인 burst를 재생산할 수 있다. 이 때문에 `retry-after`는 에러 메시지 일부가 아니라 다음 queue 파형을 만드는 제어 신호로 읽어야 한다. jitter나 tenant별 backoff는 그 뒤에 붙는 `운영 휴리스틱`이다.

- 심화 설명형 1 해설 포인트: 먼저 request count와 token backlog를 분리해 "건수는 적은데 비용만 큰가"를 본다. 다음으로 신규 prefill이 밀리는지, active decode가 오래 점유하는지 나눠 본다. 특정 tenant만 거의 전진하지 못하면 starvation을 의심하고, 그 시점에서 공정성 큐나 aging을 검토할 수 있다. 다만 그 정책 이름들은 직접 출처 사실이 아니라 운영자가 선택하는 `운영 휴리스틱`이라고 구분해야 한다.

- 심화 설명형 2 해설 포인트: A는 구현이 단순하고 평균 latency를 빠르게 개선할 수 있지만 긴 분석 요청을 더 심하게 굶길 위험이 있다. B는 prefill/decode를 분리해서 관찰하고 비교하는 구조를 제공한다. [S2] 다만 phase별 최소 보장량, tenant별 우선권, 재입장 정책은 [S2]가 직접 확정하는 내용이 아니라 운영 판단으로 남는다.
