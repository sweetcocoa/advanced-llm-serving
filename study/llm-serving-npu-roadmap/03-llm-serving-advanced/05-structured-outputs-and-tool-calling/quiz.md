# Structured Outputs and Tool Calling 퀴즈

## 객관식 1
정산 분류 시스템이 `category`, `tax_type`, `needs_review`, `explanation`을 생성한다. 이 중 structured output의 strict schema를 우선 적용해야 할 대상으로 가장 적절한 것은 무엇인가?

1. 사람이 참고만 하는 `explanation`
2. downstream 시스템이 바로 읽는 `category`, `tax_type`, `needs_review`
3. 모든 필드를 같은 강도로 묶는 것이 항상 최선이다
4. 어떤 필드도 strict schema로 묶지 않는 것이 안정적이다

## 객관식 2
tool calling 세션의 p95 latency가 높을 때 가장 먼저 의심할 원인으로 가장 적절한 것은 무엇인가?

1. 항상 GPU FLOPS 부족
2. 첫 tool 선택 실패와 중복 round 증가
3. 출력 토큰 수가 1개여도 tool calling은 무조건 느리다
4. JSON parser의 공백 처리 비용

## 객관식 3
structured output과 speculative decoding의 관계를 가장 신중하게 설명한 것은 무엇인가?

1. speculative decoding은 strict schema 구간에서도 항상 같은 비율로 이득을 준다
2. structured output을 쓰면 speculative decoding은 절대 사용할 수 없다
3. 자유 서술 구간에서는 도움이 될 수 있지만, 엄격한 제약 구간에서는 기대 효과를 따로 검증해야 한다
4. speculative decoding은 tool calling에서 tool 실행 시간을 줄여 준다

## 짧은 서술형 1
수식 `\rho_t = |V_{\mathrm{valid}}(t)| / |V|`에서 `\rho_t`가 작아질수록 얻는 이점과 잃는 것을 각각 한 문장씩 설명하라.

## 짧은 서술형 2
긴 정책 문서를 공통으로 사용하는 tool calling agent에서 `shared prefix length`를 확인하는 이유를 설명하라.

## 심화 설명형 1
환불 심사 보조 시스템이 `decision`, `risk_level`, `refund_amount`, `explanation`을 생성한다. 앞의 세 필드는 워크플로 엔진이 즉시 소비하고, `explanation`은 상담사가 읽는다.

어떤 필드를 strict schema로 두고 어떤 필드를 느슨하게 둘지 설계하라. 또한 그 선택이 `schema-valid rate`, decode 자유도, 운영 안정성에 미치는 영향을 설명하라.

## 심화 설명형 2
구매 승인 agent가 `lookup_stock`, `check_policy`, `create_order` 세 도구를 사용한다. 공통 system prompt는 길고, 첫 tool 선택 정확도가 낮으며 session p95 latency도 높다.

이 상황을 어떤 순서로 진단할지 설명하라. 답변에는 최소한 다음 항목이 포함되어야 한다.
- 어떤 지표부터 볼지
- tool taxonomy를 어떻게 점검할지
- KV cache reuse와 disaggregated prefill/decode를 언제 고려할지
- speculative decoding을 어디에 붙일지 혹은 왜 보류할지

## 정답 및 해설
### 객관식 1
정답: 2

해설: 기계가 바로 읽는 필드는 형식 실패가 곧 운영 실패로 이어지므로 strict schema 우선순위가 높다. 사람이 읽는 설명 필드까지 동일 강도로 묶으면 형식은 단단해져도 decode 자유도를 불필요하게 잃을 수 있다 [S5].

### 객관식 2
정답: 2

해설: tool calling은 한 번의 생성이 아니라 세션이다. 첫 선택이 틀리면 추가 round, 재프롬프트, validation, 외부 실행이 늘어나며 p95 latency가 빠르게 악화된다 [S6].

### 객관식 3
정답: 3

해설: speculative decoding은 자유 서술이 긴 구간에서 특히 매력적일 수 있지만 [S4], structured output처럼 허용 토큰 집합이 좁은 구간에서는 기대 이득을 별도로 검증해야 한다. 이는 [S4]와 [S5]를 함께 놓고 얻는 실무적 해석이다.

### 짧은 서술형 1
예시 답안: `\rho_t`가 작아질수록 형식 보장은 강해진다. 대신 허용 토큰 집합이 좁아져 모델의 표현 자유도와 일부 decode 효율을 희생할 수 있다.

### 짧은 서술형 2
예시 답안: 공통 prefix가 길면 같은 지시문과 정책 문서를 round마다 다시 prefill하고 있을 가능성이 있다. 따라서 shared prefix 길이를 보면 KV cache reuse [S3]나 disaggregated prefill/decode [S1][S2]로 줄일 수 있는 비용이 있는지 판단할 수 있다.

### 심화 설명형 1
예시 답안: `decision`, `risk_level`, `refund_amount`는 strict schema로 묶고 `explanation`은 길이 제한과 기본 금칙어 정도만 두는 편이 적절하다. 앞의 세 필드는 downstream 시스템이 즉시 읽으므로 schema-valid rate가 운영 안정성과 직결된다 [S5]. 반면 `explanation`까지 과도하게 제약하면 상담사가 필요한 맥락이 줄고 재시도나 수동 수정이 늘 수 있다. 즉, strict schema는 운영 안정성을 높이지만 decode 자유도는 줄이므로 필드 성격에 따라 적용 강도를 다르게 가져가야 한다.

### 심화 설명형 2
예시 답안: 첫 단계는 `schema-valid rate`, `first tool selection accuracy`, `argument validity`, `session p95 latency`를 분리해서 보는 것이다 [S5][S6]. 그다음 `lookup_stock`, `check_policy`, `create_order`의 이름과 설명이 겹쳐 첫 선택이 흔들리는지 점검한다. 공통 prompt가 길다면 shared prefix와 prefill 시간을 보고 KV cache reuse [S3]를 먼저 검토하고, prefill 부담이 decode 자원과 섞여 병목이 되면 disaggregated prefill/decode [S1][S2]를 고려한다. speculative decoding은 최종 자연어 응답이 길거나 자유 서술 구간이 분명할 때 우선 검토하고 [S4], 엄격한 함수 인자 생성 구간에는 동일한 이득을 가정하지 않는다.
