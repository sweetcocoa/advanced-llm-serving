# Speculative Decoding 퀴즈

## 객관식 1
다음 중 speculative decoding의 성능 이득을 가장 직접적으로 결정하는 조합은 무엇인가?

1. cache hit율과 eviction 주기
2. 승인율, draft 비용, target 검증 비용
3. prefill 길이와 prompt 템플릿 버전
4. retrieval 문서 개수와 namespace 개수

## 객관식 2
다음 중 speculative decoding을 가장 보수적으로 적용해야 할 가능성이 큰 route는 무엇인가?

1. 700토큰 이상의 자유 서술 보고서를 생성하는 assistant
2. 동일한 system prompt를 공유하지만 답변 형식이 완전히 자유로운 상담 bot
3. 엄격한 JSON schema와 tool calling을 함께 사용하는 주문 처리 agent
4. 공통 prefix가 길고 출력은 짧은 회의록 요약 서비스

## 객관식 3
긴 입력 30k 토큰, 출력 120토큰인 요약 서비스가 느리다. 가장 우선순위가 높은 판단으로 적절한 것은 무엇인가?

1. speculative decoding부터 최대로 켜 본다.
2. draft model을 더 크게 만들어 승인율만 올리면 된다.
3. 병목이 prefill인지 decode인지 먼저 나누고, prefill이면 prefix caching이나 disaggregated prefill/decode를 먼저 본다.
4. tool calling이 없으니 무조건 speculative decoding이 이긴다.

## 짧은 서술형 1
structured output 또는 tool calling 경로에서 speculative decoding의 승인율이 떨어질 수 있는 이유를 설명하라.

## 짧은 서술형 2
승인율이 높아 보여도 speculative decoding이 실제 속도 개선으로 이어지지 않을 수 있는 이유를 설명하라.

## 심화 설명형 1
어떤 보고서 assistant의 baseline decode 비용은 토큰당 14ms다. speculative decoding을 적용했더니 다음과 같은 측정값이 나왔다.

- `k = 4`
- draft가 제안한 4토큰의 총비용: 12ms
- target 검증 패스 1회의 비용: 18ms
- 승인율 `alpha = 0.75`

다음에 답하라.

1. README의 수식을 이용해 speculative decoding의 토큰당 기대 비용을 계산하라.
2. baseline과 비교해 속도 개선이 있는지 설명하라.
3. 같은 route에서 승인율이 0.35로 떨어지면 어떤 조정을 먼저 검토할지 쓰라.

## 심화 설명형 2
다음 세 서비스에 대해 어떤 최적화를 먼저 적용할지 설명하라.

1. 공통 system prompt와 schema가 길고 반복되며, 출력은 100토큰 안팎인 법무 질의응답 서비스
2. 입력은 짧지만 800토큰 이상 장문 서술을 자주 생성하는 경영 보고 assistant
3. 짧은 JSON 응답과 tool calling이 핵심인 환불 처리 agent

답변에는 speculative decoding, prefix caching, KV cache reuse, disaggregated prefill/decode를 모두 언급하고, 왜 우선순위가 다른지 설명하라.

## 정답 및 해설
- 객관식 1 정답: 2. speculative decoding은 승인율이 높을수록 유리하지만, draft를 추가로 돌린 비용과 target 검증 비용을 같이 봐야 실제 이득을 판단할 수 있다.
- 객관식 2 정답: 3. JSON schema와 tool calling은 형식 제약이 강해 rollback이 잦아질 수 있으므로 route별로 `k`를 줄이거나 비활성화를 검토할 가능성이 크다.
- 객관식 3 정답: 3. 긴 입력에 짧은 출력이면 prefill 병목일 가능성이 커서 speculative decoding보다 prefix caching, KV cache reuse, disaggregated prefill/decode가 먼저일 수 있다.
- 짧은 서술형 1 예시: structured output과 tool calling은 허용되는 다음 토큰의 집합이 좁다. draft가 중괄호, key, enum 값, 인자 구조를 조금만 어겨도 target이 즉시 reject해 rollback이 생기므로 승인율이 낮아질 수 있다.
- 짧은 서술형 2 예시: 승인율이 높아도 draft 자체가 너무 비싸거나 `k`가 과도하면 추가 계산 비용이 커진다. 또 workload가 decode-heavy가 아니라 prefill-heavy면 speculative decoding이 줄여 주는 구간이 작아 체감 개선이 약할 수 있다.
- 심화 설명형 1 해설: 토큰당 기대 비용은 `(12 + 18) / (0.75 * 4 + 1) = 30 / 4 = 7.5ms`다. baseline 14ms보다 작으므로 속도 개선이 있다. 승인율이 0.35로 떨어지면 같은 설정의 효율이 급락하므로 먼저 `k` 축소, route별 비활성화, structured route 분리, draft 비용 재점검을 검토하는 편이 합리적이다.
- 심화 설명형 2 해설: 1번은 반복되는 긴 prefix가 핵심이므로 prefix caching과 KV cache reuse가 우선이고, 필요하면 disaggregated prefill/decode를 본다. 2번은 decode-heavy 장문 생성이므로 speculative decoding 우선순위가 가장 높다. 3번은 tool calling과 JSON 제약이 강해 speculative decoding을 보수적으로 적용해야 하며, route별 정책 분리와 낮은 `k` 설정이 중요하다. disaggregated prefill/decode는 자원 배치 전략, KV cache reuse는 재사용 전략으로 각 workload의 병목에 따라 보조적으로 결합한다.
