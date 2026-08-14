# GPU vs NPU Mental Model 퀴즈

## 객관식 1
노트북 회의 요약기에서 다음 중 이 챕터의 관점에 가장 맞는 partition은 무엇인가?

1. `512-token chunk 요약 prefill`은 NPU 후보로 두고, `최종 자유 형식 decode`는 GPU에 남긴다.
2. `최종 자유 형식 decode`의 각 토큰마다 projection 일부를 NPU로 보냈다가 다시 GPU로 받는다.
3. 지원 범위를 확인하지 않은 채 모든 블록을 우선 NPU로 옮기고, 실패하면 중간마다 fallback을 붙인다.
4. 장치 경계가 많을수록 세밀한 최적화가 가능하니 최대한 잘게 나눈다.

## 객관식 2
다음 중 `[S3]`와 `[S4]`의 관계를 가장 정확히 설명한 것은 무엇인가?

1. `[S3]`와 `[S4]`는 둘 다 GPU가 항상 기준선이라고 직접 선언한다.
2. `[S3]`는 serving runtime 맥락을, `[S4]`는 특정 NPU runtime 지원 제약 사례를 제공하며, 그 위의 partition 규칙은 강의 해석이다.
3. `[S4]`는 모든 NPU의 공통 성질을 증명하고, `[S3]`는 hybrid 증가 추세를 통계로 제시한다.
4. `[S3]`와 `[S4]`는 둘 다 장치 성능표만 제공하므로 partition 판단에는 거의 도움이 없다.

## 객관식 3
다음 중 `나쁜 cut`의 전형적인 신호는 무엇인가?

1. 안정적인 shape를 가진 블록을 decode loop 바깥에서 한 번만 handoff한다.
2. unsupported op가 나오면 일찍 escape한 뒤 남은 경로를 한 runtime에 둔다.
3. 작은 블록 이득을 위해 decode loop 안쪽에서 장치를 토큰마다 왕복한다.
4. shape 안정성, escape path, handoff 횟수를 표로 적고 비교한다.

## 짧은 서술형 1
왜 `final decode`를 GPU에 남기고, `stable prefill block`만 NPU 후보로 두는 해석이 이 챕터에서 더 안전한가? 답변에는 `shape 안정성`, `decode loop`, `handoff`를 반드시 포함하라.

## 짧은 서술형 2
`[S4]`를 사용할 때 왜 "NPU 일반론"이 아니라 "특정 runtime 지원/제약 사례"로만 다뤄야 하는지 설명하라. 답변에는 `release note`, `지원 범위`, `강의 해석`을 반드시 포함하라.

## 심화 설명형 1
다음 상황에서 partition 결정표를 어떻게 작성할지 설명하라.

- 환경: 노트북 회의 요약기
- 파이프라인: `512-token chunk` 반복 요약 후, 최종 자유 형식 요약 생성
- 목표: 발열을 줄이되, 요약 문장 생성 latency가 튀지 않게 유지

답변 조건:

- `chunk 요약 prefill`, `최종 decode`, `unsupported-op escape path`를 모두 다룰 것
- 어디서 handoff를 허용하고 어디서는 피해야 하는지 말할 것
- `[S3]`의 runtime 맥락과 `[S4]`의 지원 제약 사례가 각각 어디에 쓰이는지 구분할 것

## 심화 설명형 2
다중 사용자 서버 챗봇에서 "GPU를 기준선으로 두고 시작한다"는 말을 어떻게 방어적으로 설명할 수 있는가? 아래 조건을 만족해 답하라.

- 이 문장이 `출처의 직접 사실`이 아니라 `강의 해석`임을 먼저 밝힐 것
- `요청 길이 변동`, `decode 종료 시점`, `경계 축소`를 포함할 것
- NPU를 완전히 배제하지 말고, 어떤 블록이 생기면 다시 검토할지 적을 것

## 정답 및 해설
- 객관식 1 정답: 1. 이 챕터의 핵심은 안정 블록만 분리하고 가변 decode는 한 runtime에 오래 남기는 것이다. `512-token chunk`처럼 shape가 반복되는 prefill은 후보가 될 수 있지만, 자유 형식 decode를 토큰마다 왕복시키는 것은 나쁜 cut이다.
- 객관식 2 정답: 2. [S3]는 serving runtime 맥락, [S4]는 특정 NPU runtime의 지원 제약 사례를 준다. "stable block만 분리하자" 같은 partition 규칙은 그 위에서 세운 강의 해석이다.
- 객관식 3 정답: 3. decode loop 안쪽에서 장치를 토큰마다 왕복하면 작은 블록 이득보다 handoff 비용이 더 커질 수 있다. 이 챕터는 이런 경우를 대표적인 bad cut으로 본다.
- 짧은 서술형 1 해설 포인트: stable prefill block은 shape 안정성이 높아 NPU 후보가 될 수 있지만, final decode는 loop 내부에서 길이와 종료 시점이 계속 흔들린다. 그래서 decode를 한 runtime에 두어 handoff를 줄이는 해석이 더 안전하다. 여기서 중요한 비교 축은 장치 이름이 아니라 `shape 안정성`과 `handoff 횟수`다.
- 짧은 서술형 2 해설 포인트: [S4]는 release note이므로 특정 runtime의 지원 범위와 제약이 어떻게 관리되는지 보여 주는 사례다. 이 문서 하나로 모든 NPU의 일반 성질을 단정하면 과장이다. 따라서 [S4]는 "escape path를 미리 적어 두라"는 강의 해석을 보조하는 수준으로만 써야 한다.
- 심화 설명형 1 채점 포인트: `chunk 요약 prefill`은 shape 안정성이 높으므로 NPU 후보로 둘 수 있다. `최종 decode`는 가변적이어서 GPU에 남기는 편이 안전하다. unsupported op가 나오면 일찍 escape해 남은 경로를 한 runtime에서 유지해야 하며, handoff는 decode 시작 전처럼 횟수가 적은 지점에서만 허용하는 답변이 적절하다. [S3]는 runtime 맥락, [S4]는 지원 제약 사례로 구분해 써야 한다.
- 심화 설명형 2 채점 포인트: "GPU를 기준선으로 둔다"는 말은 [S3]가 직접 선언한 문장이 아니라, 요청 길이 변동과 decode 종료 시점 변동이 큰 환경에서 경계를 성급히 늘리지 않기 위한 강의 해석이라고 밝혀야 한다. 동시에 shape가 안정적인 반복 블록이 식별되면 그때 NPU 후보를 다시 검토해야 한다고 적어야 균형이 맞다.
