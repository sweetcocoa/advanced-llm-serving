# Ryzen AI OGA 퀴즈

## 객관식 1

AMD Ryzen AI OGA 챕터를 읽을 때 가장 먼저 붙잡아야 하는 표현은 무엇인가?

1. 단일 NPU 최대 활용
2. Hybrid On-Device GenAI workflow
3. Windows 앱 배포 자동화
4. provider 옵션 최소화

## 객관식 2

다음 중 `GPU prefill -> NPU decode` split을 먼저 검토할 조건으로 가장 적절한 것은 무엇인가?

1. 답변이 매우 짧고 handoff가 여러 번 생길 때
2. 첫 토큰이 중요하고 prompt가 길며, handoff를 한 번 수준으로 묶을 수 있을 때
3. provider 로그를 전혀 볼 수 없을 때
4. OS 통합만으로 장치 경계가 완전히 사라질 때

## 객관식 3

다음 중 Ryzen AI OGA 챕터에서 `전력 효율과 개발 복잡도`를 함께 설명하는 문장으로 가장 적절한 것은 무엇인가?

1. NPU 사용 시간이 길수록 회귀 검증 비용은 자동으로 줄어든다.
2. 긴 decode 구간은 NPU 이점을 만들 수 있지만, artifact 준비와 fallback 시험이 늘면 운영 비용도 함께 커질 수 있다.
3. Windows ML을 쓰면 handoff 비용은 항상 0이 된다.
4. Qualcomm AI Hub를 보면 AMD hybrid 설계를 따로 볼 필요가 없어진다.

## 짧은 서술형 1

`G_split`이 0 이하가 되는 상황을 2~3문장으로 설명하라.

## 짧은 서술형 2

오프라인 영상 자막 초안 생성기와 배터리 우선 개인 코파일럿 중 하나를 골라, 왜 split 후보가 달라지는지 설명하라.

## 심화 설명형 1

Ryzen AI OGA 관점에서 `CPU orchestration - GPU prefill - NPU decode`라는 학습용 split을 언제 먼저 검토하고, 언제 단일 경로로 되돌릴지 설명하라. 답변에는 `first token`, `긴 decode`, `handoff`, `운영 복잡도`가 모두 포함되어야 한다.

## 심화 설명형 2

AMD 문서 S1을 중심에 두고, S2, S3, S4를 각각 어떤 보조 비교 축으로 써야 하는지 설명하라. 답변에는 `artifact`, `provider`, `OS 통합`이 모두 포함되어야 한다.

## 정답 및 해설

- 객관식 1 정답: 2. AMD 문서 제목 자체가 `Hybrid On-Device GenAI workflow`이므로, 이 챕터의 중심도 단일 장치 최대 활용이 아니라 hybrid 설계 문제다 [S1].
- 객관식 2 정답: 2. 첫 토큰이 중요하고 prompt가 길며 handoff를 한 번 수준으로 묶을 수 있을 때는 GPU prefill 뒤에 NPU decode를 붙이는 split을 먼저 검토할 수 있다 [합성] [S1].
- 객관식 3 정답: 2. 긴 decode는 NPU 전력 이점을 노릴 수 있지만, artifact 준비 [S2], provider debug [S3], OS 수준 제품 검증 [S4]이 함께 붙으므로 운영 복잡도도 같이 커질 수 있다 [합성] [S1][S2][S3][S4].
- 짧은 서술형 1 예시: `G_split = (T_single - T_split_compute) - T_handoff`이므로, handoff 비용이 계산 절감분보다 크면 `G_split`은 0 이하가 된다. 짧은 응답이나 잦은 장치 전환에서는 hybrid를 써도 체감 이득이 거의 남지 않을 수 있다 [합성] [S1].
- 짧은 서술형 2 예시: 오프라인 영상 자막 초안 생성기는 긴 입력 구간의 첫 반응이 중요하므로 GPU prefill을 먼저 가설로 두기 쉽다. 반면 배터리 우선 개인 코파일럿은 긴 세션의 누적 에너지와 발열이 중요하므로 NPU decode 비중을 더 길게 검토하게 된다 [합성] [S1].
- 심화 설명형 1 해설 포인트: first token이 중요하고 prompt가 길면 `GPU prefill -> NPU decode` split을 후보에 올릴 수 있다 [합성] [S1]. 하지만 긴 decode가 실제로 길지 않거나 handoff가 여러 번 생기면 `T_handoff`가 이득을 잠식하므로 단일 GPU나 단일 NPU 경로로 되돌려야 한다 [합성] [S1]. 또한 artifact 준비 [S2], provider debug [S3], 제품 회귀 검증 [S4]을 감당할 수 있을 때만 hybrid가 운영상 성립한다.
- 심화 설명형 2 해설 포인트: S1은 AMD의 hybrid workflow 자체를 설명하는 중심 자료다 [S1]. S2는 hybrid 설계를 실제 배포로 옮길 때 어떤 artifact 준비가 필요한지 떠올리게 하고 [S2], S3는 provider와 backend 경계 및 fallback 해석을 보조하며 [S3], S4는 Windows 제품화 단계에서 OS 통합 표면을 어떻게 설명할지 보조한다 [S4].
