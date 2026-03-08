# NPU-Friendly Quantization 퀴즈

## 객관식 1
다음 중 이 챕터가 말하는 `NPU-friendly quantization`을 가장 정확히 설명한 것은 무엇인가?
1. 가능한 한 가장 낮은 bit-width를 고르는 일
2. 정확도 손실이 가장 작은 양자화 기법을 고르는 일
3. quantized artifact가 runtime/import 경로와 맞도록 표현 형식과 calibration을 고정하는 일
4. weight만 양자화하고 activation은 항상 FP16으로 두는 일

## 객관식 2
`QOperator`와 `QDQ`를 비교할 때 가장 먼저 확인해야 할 질문으로 적절한 것은 무엇인가?
1. 파일 크기가 더 작은 쪽이 무엇인가
2. runtime이 어떤 형식을 더 안정적으로 읽고 partition하는가
3. zero-point를 전혀 사용하지 않는 쪽이 무엇인가
4. calibration dataset이 전혀 필요 없는 쪽이 무엇인가

## 객관식 3
다음 중 calibration을 가장 적절하게 설명한 것은 무엇인가?
1. 학습을 다시 하지 않기 위해 optimizer state를 삭제하는 과정
2. scale과 zero-point를 고정하기 위한 기준 데이터를 정해 quantized artifact를 재현 가능하게 만드는 과정
3. runtime이 unsupported op를 자동으로 제거하게 만드는 과정
4. execution provider가 NPU를 탐지하는 과정

## 짧은 서술형 1
같은 Int8 모델이라도 `QNN EP` 경로와 `OpenVINO NPU` 경로에서 먼저 확인할 항목이 달라지는 이유를 3문장 이내로 설명하라.

## 짧은 서술형 2
다음 식이 왜 이 챕터에서 중요했는지 설명하라.

$$
T_{\mathrm{total}} =
T_{\mathrm{int8\_compute}}
+ N_{\mathrm{boundary}} \cdot T_{\mathrm{handoff}}
+ N_{\mathrm{requant}} \cdot T_{\mathrm{requant}}
$$

## 심화 설명형 1
한 팀이 ORT로 정적 양자화를 수행한 뒤 accuracy drop이 작다는 이유로 바로 배포를 진행했다. 그런데 QNN EP에서 기대보다 NPU offload가 짧고 latency도 개선되지 않았다.

이 상황에서 확인해야 할 순서를 다음 키워드를 모두 사용해 설명하라.
- `QDQ/QOperator`
- calibration dataset
- partition
- boundary 또는 requant

## 심화 설명형 2
다음 두 선택을 비교해 설명하라.
- per-tensor vs per-channel
- symmetric vs asymmetric

답안에는 반드시 다음 두 요소를 포함하라.
- 왜 이것이 `품질 문제`이면서 동시에 `import 문제`인지
- 왜 이 챕터가 세부 지원표를 단정하지 않고 artifact 관점으로 설명하는지

## 정답 및 해설
- 객관식 1 정답: 3. 이 챕터의 핵심은 bit-width 자체보다 quantized artifact가 runtime/import 경로와 맞는지다. 표현 형식과 calibration 고정이 함께 봐야 할 대상이다.
- 객관식 2 정답: 2. `QOperator`와 `QDQ`는 표기 차이가 아니라 graph 구조 차이이므로, 실제 runtime이 어떤 형식을 읽고 partition하는지부터 확인해야 한다.
- 객관식 3 정답: 2. calibration은 activation range를 수집해 scale과 zero-point를 정하는 과정이며, 결과적으로 quantized artifact의 재현성에도 직접 연결된다.
- 짧은 서술형 1 예시: QNN EP 경로에서는 quantized artifact가 실제로 어떤 partition으로 잡히는지가 먼저 중요하다. OpenVINO NPU 경로에서는 NNCF의 calibration 및 `target_device` 설정과 NPU device 문맥이 더 먼저 보인다. 둘 다 Int8일 수 있어도 runtime이 artifact를 읽는 방식이 다르기 때문에 질문 순서가 달라진다.
- 짧은 서술형 2 예시: 이 식은 Int8 계산이 빨라 보여도 경계 handoff와 requantization이 많으면 전체 latency 이득이 사라질 수 있음을 보여 준다. 그래서 양자화 품질이 좋아도 배포 성능은 나쁠 수 있다는 점을 한 줄로 정리해 준다. 이 챕터가 `좋은 양자화`와 `배포되는 양자화`를 구분한 이유가 여기 있다.
- 심화 설명형 1 해설 포인트: 먼저 export 결과가 `QDQ`인지 `QOperator`인지 확인해 graph 표현을 고정한다. 다음으로 calibration dataset과 설정을 확인해 scale/zero-point가 재현 가능한 artifact였는지 본다. 그 뒤 QNN EP partition이 어디서 끊기는지 보고, 마지막으로 boundary handoff 또는 requant 구간이 latency를 잡아먹는지 확인한다. accuracy drop이 작았다는 사실만으로 deployability를 보장할 수 없다는 결론이 핵심이다.
- 심화 설명형 2 해설 포인트: per-channel은 per-tensor보다 더 세밀하게 scale을 둘 수 있어 품질 보존 여지가 있지만 artifact 구조와 import 해석이 더 복잡해질 수 있다. symmetric와 asymmetric는 zero-point 처리 방식 차이이며, 표현 범위와 파라미터 구조에 영향을 준다. 이 챕터가 세부 지원표를 단정하지 않는 이유는 제공된 근거가 `지원 범위 표`보다 `QDQ/QOperator`, calibration, target device, runtime import 경로 설명`에 집중돼 있기 때문이다. 따라서 실무자는 지원표를 추정하기보다 먼저 artifact 형식과 import 경로를 확인해야 한다.
