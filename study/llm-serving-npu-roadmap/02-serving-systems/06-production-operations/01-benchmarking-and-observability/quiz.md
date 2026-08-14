# Benchmarking and Observability 퀴즈

## 객관식 1

다음 중 observability dashboard의 질문으로 가장 적절한 것은 무엇인가?

1. 고정된 입력 길이와 동시성에서 설정 A와 B 중 어느 쪽이 더 빠른가?
2. 오늘 오후 배포 이후 왜 `TTFT p95`가 갑자기 상승했는가?
3. 같은 synthetic workload에서 어느 엔진이 더 높은 throughput을 내는가?
4. warm-up 조건을 같게 둘 때 어떤 설정이 더 안정적인가?

## 객관식 2

이 챕터에서 TensorRT-LLM의 disaggregated serving을 직접 근거로 끌어오는 이유로 가장 적절한 것은 무엇인가?

1. prefill과 decode를 합쳐서 읽어도 충분하다는 점을 보여 주기 위해
2. prefill과 decode를 다른 운영 단위와 다른 계측 대상으로 볼 수 있음을 보여 주기 위해
3. observability 없이 benchmark만으로 운영 원인을 찾을 수 있음을 보여 주기 위해
4. structured output 지표의 표준값을 정의하기 위해

## 객관식 3

다음 중 이 챕터에서 공식 출처 기반 주장이라기보다 교수자 운영 휴리스틱으로 명시된 항목은 무엇인가?

1. disaggregated serving은 prefill/decode 분리 관점을 제공한다
2. benchmark와 observability는 같은 질문에 답하지 않는다
3. structured output 서비스에서는 `schema-valid rate`를 속도 그래프 옆에 두는 편이 안전하다
4. workload mix를 드러내지 않는 평균값은 배포 판단을 흐릴 수 있다

## 짧은 서술형 1

benchmark가 현실을 일부러 단순화해야 하는 이유를 2~3문장으로 설명하라.

## 짧은 서술형 2

`TTFT = T_queue + T_prefill + T_first-token` 분해식이 운영 디버깅에서 왜 유용한지 설명하라.

## 심화 설명형 1

짧은 채팅 요청과 긴 문서 요약 요청이 함께 들어오는 서비스를 맡았다고 하자. benchmark bucket을 어떻게 나누고, 각 bucket에서 어떤 값을 먼저 볼지 설명하라.

## 심화 설명형 2

배포 직후 사용자 불만이 늘었다. `TTFT p95`, queue wait, phase 분리, replay benchmark를 포함해 진단 순서를 서술하라. structured output 실패를 함께 의심해야 하는 상황이라면 어떤 값이 공식 출처 기반이 아니라 운영 휴리스틱인지도 구분해서 적어라.

## 정답 및 해설

- 객관식 1 정답: 2. observability는 "왜 지금 나빠졌는가"를 묻는 도구다. 1, 3, 4는 조건을 고정한 비교 질문이라 benchmark scorecard 쪽에 가깝다.
- 객관식 2 정답: 2. [S2]는 prefill과 decode를 다른 운영 단위로 볼 수 있다는 점을 이 챕터의 직접 근거로 제공한다. 그래서 phase별 benchmark와 운영 계측을 따로 설계해야 한다는 설명이 가능해진다.
- 객관식 3 정답: 3. `schema-valid rate`를 structured output 운영 지표로 두는 설명은 이 챕터에서 교수자 운영 휴리스틱으로 명시했다. 반면 1은 [S2]의 직접 근거, 2와 4는 챕터의 핵심 개념 정리다.
- 짧은 서술형 1 예시: benchmark는 입력 길이, 출력 길이, 동시성 같은 조건을 고정해야 설정 간 차이를 선명하게 비교할 수 있다. 운영 현실을 그대로 복사하면 숫자는 많아지지만 어떤 변화가 설정 때문인지 workload mix 때문인지 흐려지기 쉽다.
- 짧은 서술형 2 예시: 같은 `TTFT` 상승이라도 queue가 원인인지, 긴 입력 때문에 prefill이 늘어난 것인지, first-token 직전 단계가 막힌 것인지에 따라 대응이 달라진다. 분해식은 latency를 한 숫자로 보지 않고 병목 가설을 단계별로 나누게 해 준다.
- 심화 설명형 1 해설 포인트: 최소한 `짧은 채팅 bucket`과 `긴 문서 요약 bucket`은 분리해야 한다. 짧은 채팅에서는 `TTFT`와 queue 대기가 먼저 중요하고, 긴 문서 요약에서는 prefill 비중과 phase 분리가 더 중요하다. 평균 하나로 합치면 workload mix 변화가 설정 효과처럼 보일 수 있다.
- 심화 설명형 2 해설 포인트: 먼저 불만이 첫 응답 지연인지, 중간 생성 속도 저하인지, 형식 실패인지 분류한다. 그 다음 `TTFT p95`, queue wait, prefill/decode 분리값을 보고 증상을 특정한 뒤, 실제 workload mix를 반영한 replay benchmark로 재현한다. structured output 실패를 함께 의심한다면 `schema-valid rate`와 retry 비율을 볼 수 있지만, 이 부분은 [S1], [S2]의 직접 근거가 아니라 운영 휴리스틱으로 적는 것이 맞다.
