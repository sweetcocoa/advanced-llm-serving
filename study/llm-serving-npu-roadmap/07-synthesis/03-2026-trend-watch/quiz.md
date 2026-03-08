# 2026 Trend Watch 퀴즈

## 객관식 1
이 챕터에서 `structured outputs`를 다루는 방식으로 가장 정확한 것은 무엇인가?

1. 2026년의 대표 인프라 기술이므로 `주류 진입`으로 분류한다.
2. 현재 소스 집합이 직접 다루지 않으므로 중요하지 않은 주제라고 제외한다.
3. 현재 소스 범위에서는 `비분류인 제품 요구 신호`로 두고, validation path와 routing 비용을 늘리는 압력으로 읽는다.
4. on-device 전용 최적화 기술로 분류한다.

## 객관식 2
다음 중 cloud 쪽에서 `disaggregated serving`이 2026년 `주류 진입`에 더 가깝다고 보는 이유로 가장 적절한 것은 무엇인가?

1. 어떤 workload에서도 speculative decoding보다 항상 더 빠르기 때문이다.
2. 긴 prompt workload에서 prefill 병목이 명확하고, prefill/decode 분리를 공식 기능으로 다루기 때문이다.
3. 구조화 출력이 필요하면 무조건 disaggregated serving이 필요하기 때문이다.
4. on-device runtime보다 구현이 단순하기 때문이다.

## 객관식 3
다음 중 on-device 경로를 아직 `실험 신호`로 봐야 하는 사례는 무엇인가?

1. provider coverage와 fallback 로그를 지속적으로 수집하고 있다.
2. local 처리와 cloud escalation 조건이 문서화돼 있다.
3. 한 기기 데모만 성공했지만 CPU/GPU fallback 비율은 모른 채 범용 배포를 약속한다.
4. hybrid workflow에서 validator와 repair 경계를 따로 설계했다.

## 짧은 서술형 1
`T_{\mathrm{safe\_resp}} = T_{\mathrm{prefill}} + T_{\mathrm{decode}} + T_{\mathrm{validate}} + T_{\mathrm{repair}} + T_{\mathrm{route}}` 식을 이용해, structured outputs가 왜 latency budget과 hybrid/cloud 선택 압력을 동시에 높이는지 2~3문장으로 설명하라.

## 짧은 서술형 2
이 챕터의 on-device 체크리스트에서 `채택 신호` 2개와 `실험 신호` 1개를 골라 적고, 왜 그렇게 나누는지 짧게 설명하라.

## 심화 설명형 1
다음 서비스는 긴 계약서를 읽고 짧은 JSON 결과를 돌려주는 `규정 점검 API`다. 아래 질문에 답하라.

- 왜 structured outputs 요구가 있어도 speculative decoding을 무조건 1순위로 두면 안 되는가?
- disaggregated serving, validator, repair path를 어떤 순서로 점검해야 하는가?
- 이 서비스에서 `주류 진입` 기술과 `선별 도입` 기술을 각각 하나씩 고르고 근거를 설명하라.

## 심화 설명형 2
다음 서비스는 AI PC에서 현장 점검 체크리스트를 생성하고, 복잡한 복구 조언만 cloud로 보내는 `현장 점검 도우미`다. 아래 질문에 답하라.

- 왜 `NPU 지원` 문구만으로는 채택 결정을 내릴 수 없는가?
- provider coverage, fallback 로그, cloud escalation 규칙이 왜 2026년 `채택 신호`인지 설명하라.
- structured outputs 요구가 붙을 때 local validator와 cloud repair의 경계를 어떻게 설계해야 하는지 제안하라.

## 정답 및 해설
- 객관식 1 정답: 3. 이번 챕터는 structured outputs를 직접 문서화한 출처가 없으므로 인프라 maturity를 판정하지 않고, validation과 routing 비용을 늘리는 제품 요구 신호로 다룬다.
- 객관식 2 정답: 2. vLLM 문서는 prefill/decode 분리를 공식 기능으로 제시하고 있고, 긴 prompt workload에서 병목 위치가 분명하기 때문에 `주류 진입`으로 분류했다. [S1]
- 객관식 3 정답: 3. 한 기기 데모 성공만으로는 범용 채택 신호가 되지 않는다. 2026년 on-device 채택 여부는 coverage와 fallback 가시성까지 확인해야 한다. [S3][S4][S6]
- 짧은 서술형 1 예시: structured outputs가 들어오면 생성 시간만으로 끝나지 않고 schema 검증과 오류 복구 시간이 추가된다. local 경로에서 형식 검증이나 repair가 불안정하면 cloud나 다른 장치 경로로 우회해야 하므로 `T_{\mathrm{route}}`가 커지고, 그 결과 latency budget과 hybrid/cloud 선택 압력이 함께 올라간다. 이 해석은 prefill/decode 병목 [S1][S2]과 fallback/offload 경계 [S3][S4][S5][S6]를 연결한 수업용 추론이다.
- 짧은 서술형 2 예시: 채택 신호로는 `provider/runtime surface가 공식 문서로 정리돼 있다`, `fallback 로그를 계측할 수 있다`를 들 수 있다 [S3][S4][S6]. 실험 신호로는 `한 SKU 데모 성공을 범용 호환성으로 확대 해석한다`를 들 수 있다. 전자는 반복 가능한 운영 절차를 만들 수 있지만, 후자는 재현성과 관측성이 부족하다.
- 심화 설명형 1 해설 포인트: 긴 계약서와 짧은 JSON 응답이라면 먼저 prefill 병목과 TTFT를 본다 [S1]. structured outputs가 있어도 validator와 repair path 비용이 decode보다 큰지 따져야 하며, speculative decoding은 decode 병목이 실제로 큰 경우에만 `선별 도입`으로 검토한다 [S2]. 이 사례에서 `주류 진입` 기술은 disaggregated serving, `선별 도입` 기술은 speculative decoding이 적절하다.
- 심화 설명형 2 해설 포인트: `NPU 지원`은 경로 존재를 뜻할 뿐, 실제 coverage와 fallback 비율을 보장하지 않는다 [S3][S6]. provider coverage, fallback 로그, cloud escalation 규칙이 있어야 local 경로와 hybrid 경로를 반복 가능하게 운영할 수 있다 [S4][S5]. structured outputs가 붙으면 local validator를 어디까지 둘지, 실패 시 cloud repair를 어떤 조건에서만 부를지 먼저 정해야 privacy와 latency 약속을 동시에 지킬 수 있다.
