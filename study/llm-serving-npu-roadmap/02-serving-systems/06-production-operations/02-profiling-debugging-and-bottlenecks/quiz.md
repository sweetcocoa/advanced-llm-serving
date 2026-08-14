# Profiling, Debugging, and Bottlenecks 퀴즈

## 객관식 1
NPU workload가 기대보다 느릴 때, 이 챕터가 가장 먼저 확인하라고 권하는 대상은 무엇인가?

1. 장치 사용률 숫자
2. partition과 host/device 경계
3. 모델 파라미터 개수
4. 배치 크기만 키우는 실험

## 객관식 2
CPU fallback을 막았더니 세션이 아예 성립하지 않았다. 이 상황에 대한 가장 적절한 해석은 무엇인가?

1. profiler 파일이 손상되었다
2. tracing을 더 오래 켜 두면 자동으로 해결된다
3. graph 또는 세션 성립 조건을 먼저 점검해야 한다
4. request 수를 늘리면 세션이 살아난다

## 객관식 3
OpenVINO 기반 측정에서 benchmark 결과는 안정적이지만 실제 서비스 latency는 요청 패턴에 따라 크게 흔들린다. 가장 먼저 분리해서 봐야 할 것은 무엇인가?

1. request 수와 performance hint 같은 request-level 설정
2. 이미지 파일 해상도
3. 모델의 학습 epoch 수
4. 운영체제 테마 설정

## 객관식 4
다음 두 상황에 대해 가장 적절한 "첫 도구 선택" 조합은 무엇인가?

- 상황 A: trace에는 NPU partition이 보이지만 p95가 안 내려가고, 로그에는 unsupported op 단서가 남아 있다.
- 상황 B: OpenVINO benchmark layer 합계는 비슷한데 multi-request에서만 p95가 흔들린다.

1. A는 utilization 대시보드만 보고, B는 모델 파라미터 수를 다시 센다
2. A는 ORT trace와 logging/tracing을 먼저 보고, B는 request 수와 performance hint를 먼저 분리해 본다
3. A는 request 수부터 늘리고, B는 unsupported op 로그부터 찾는다
4. A와 B 모두 layer별 가장 긴 막대 하나만 찾는다

## 짧은 서술형 1
왜 이 챕터는 긴 operator 하나를 찾기보다 짧은 host/device 경계의 반복을 먼저 보라고 하는가? 3문장 이내로 설명하라.

## 짧은 서술형 2
logging/tracing이 profiling trace와 다른 역할을 하는 이유를 3문장 이내로 설명하라.

## 심화 설명형 1
다음 상황을 가정하라.

- profiling trace에는 NPU partition이 여러 개 보인다.
- end-to-end latency는 기대보다 거의 줄지 않았다.
- 로그에는 fallback 관련 단서가 남아 있다.

이때 어떤 순서로 병목을 좁혀 갈 것인가? 답변에는 반드시 다음을 포함하라.
- partition 확인 이후 왜 fallback 이유를 붙여 봐야 하는가
- cast/QDQ와 transfer를 언제 의심해야 하는가
- 바로 취할 다음 액션 2개 이상

## 심화 설명형 2
다음 상황을 가정하라.

- OpenVINO benchmark 결과는 나쁘지 않다.
- 첫 요청은 느리지만 반복 요청은 비교적 안정적이다.
- request 수와 performance hint를 조정할 수 있다.

이때 layer 병목과 request-level 병목을 어떻게 분리할 것인가? 답변에는 반드시 다음을 포함하라.
- 왜 benchmark 숫자만으로 결론 내리면 안 되는가
- 첫 요청 비용과 반복 요청 병목을 어떻게 나눌 것인가
- request 설정 조정으로 넘어가는 기준

## 정답 및 해설
- 객관식 1 정답: 2. 이 챕터의 출발점은 `NPU가 켜졌는가`가 아니라 `어디서 실행이 끊겼는가`다. partition과 host/device 경계를 먼저 봐야 이후의 fallback, cast, transfer 해석이 가능하다.
- 객관식 2 정답: 3. CPU fallback을 막자 세션이 안 서는 것은 profiler 해석 문제가 아니라 graph 또는 세션 성립 조건 문제로 보는 편이 맞다. unsupported op, shape 조건, export artifact를 먼저 확인해야 한다.
- 객관식 3 정답: 1. benchmark 결과가 괜찮아도 request 수와 performance hint에 따라 서비스 latency 양상은 달라질 수 있다. 따라서 layer 시간과 request orchestration을 먼저 분리해서 봐야 한다.
- 객관식 4 정답: 2. 상황 A는 ORT trace에서 host gap을 찾고 logging/tracing으로 unsupported op 또는 fallback 이유를 붙여야 한다. 상황 B는 layer 합계가 비슷한데 multi-request에서만 흔들리므로 request 수와 performance hint, first-request 비용을 먼저 분리해서 봐야 한다.
- 짧은 서술형 1 예시: NPU workload에서는 긴 operator 하나보다 host/device 경계가 여러 번 반복되는 쪽이 더 큰 비용이 될 수 있다. cast, fallback, transfer가 경계마다 붙으면 device 계산 시간이 짧아도 전체 요청 시간은 길어진다. 그래서 이 챕터는 막대 길이보다 실행이 얼마나 쪼개졌는지를 먼저 본다.
- 짧은 서술형 2 예시: profiling trace는 어떤 구간이 언제 실행됐는지 보여 주는 도구다. 반면 logging/tracing은 fallback 이유, 세션 맥락, provider 관련 단서를 붙여 준다. 둘을 같이 봐야 `무슨 일이 있었는지`와 `왜 그랬는지`를 연결할 수 있다.
- 심화 설명형 1 해설 포인트: 먼저 trace에서 host와 device 구간이 어떻게 섞였는지 확인한다. 그다음 logging/tracing으로 그 host 구간이 왜 남았는지, 즉 fallback 이유를 붙여 본다. 이후 cast/QDQ가 경계 앞뒤에 반복되는지, transfer/sync가 누적되는지 본다. 다음 액션으로는 unsupported op 수정, precision 경계 축소, partition 재검토, 세션 옵션 재검증 등이 가능하다.
- 심화 설명형 2 해설 포인트: benchmark 숫자는 출발점일 뿐 서비스 패턴 전체를 대신 설명하지 않는다. 첫 요청만 느리다면 세션 준비나 초기화 비용을 분리하고, steady-state가 안정적인지 따로 본다. 그다음 multi-request에서만 p95가 흔들리면 request 수와 performance hint 조정으로 넘어가고, layer counter 합계가 계속 튄다면 그때 layer 병목을 더 깊게 본다.
