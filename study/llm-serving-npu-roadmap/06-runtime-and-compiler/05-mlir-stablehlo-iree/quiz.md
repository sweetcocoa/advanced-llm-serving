# MLIR, StableHLO, IREE 퀴즈

## 객관식 1
다음 중 README의 `R_lowering` 설명과 가장 잘 맞는 것은 무엇인가?

1. ONNX importer가 모델 파일을 읽었으면 `R_lowering`은 자동으로 1이 된다.
2. `R_lowering`은 다음 단계가 요구하는 legality 조건을 얼마나 실제로 충족했는지 보는 학습용 비율이다.
3. `R_lowering`은 StableHLO를 쓰는 순간 항상 높아진다.
4. `R_lowering`은 runtime latency 평균만으로 계산된다.

## 객관식 2
다음 중 README가 설명한 StableHLO의 위치를 가장 정확히 고른 것은 무엇인가?

1. StableHLO는 높은 수준 연산 의미를 유지하는 데 유용하지만, 자동 최적 codegen을 보장하지는 않는다.
2. StableHLO는 ONNX와 동일하므로 importer만 바꾸면 된다.
3. StableHLO를 쓰면 MLIR dialect conversion은 건너뛸 수 있다.
4. StableHLO를 쓰면 IREE runtime artifact 확인은 불필요해진다.

## 객관식 3
README가 제안한 디버깅 순서로 가장 적절한 것은 무엇인가?

1. backend tuning -> dispatch 확인 -> legality 확인 -> 입력 IR 의미 확인
2. 입력 IR 의미 확인 -> MLIR legality 확인 -> IREE dispatch/artifact 확인 -> backend tuning
3. ONNX import 성공 확인 -> 곧바로 latency 측정 -> StableHLO 도입 -> artifact 확인
4. runtime UI 로그 확인 -> logo 이미지 확인 -> compiler pass 이름 정렬 -> backend 변경

## 짧은 서술형 1
`ONNX import 성공`이 왜 `deployment 준비 완료`를 뜻하지 않는지, `R_lowering`을 사용해 2~3문장으로 설명하라.

## 짧은 서술형 2
더 이른 decomposition이 왜 항상 좋은 선택이 아닌지, `G_net`과 `C_dispatch`를 포함해 설명하라.

## 심화 설명형 1
어떤 팀이 ONNX 모델을 IREE에 넣었고 importer는 통과했지만, target compile 단계에서 막혔다. README의 순서를 따라 무엇을 어떤 순서로 확인할지 서술하라. 답안에는 `입력 의미`, `dialect legality`, `dispatch/artifact`, `backend tuning` 네 표현을 모두 포함하라.

## 심화 설명형 2
다음 두 선택을 비교해 어느 쪽이 더 나을지 설명하라.

1. StableHLO 수준에서 큰 연산 블록을 조금 더 유지한 뒤 나중에 target lowering을 본다.
2. 초기에 더 잘게 decomposition해서 작은 연산 조각들로 빠르게 내린다.

답안에는 `B_backend`, `C_dispatch`, `C_boundary`, `자동 최적화 보장 아님`을 모두 포함하라.

## 정답 및 해설

- 객관식 1 정답: 2. `R_lowering`은 importer 성공 여부가 아니라, 다음 단계가 요구하는 legality 조건을 얼마나 실제로 충족했는지를 설명하기 위해 README가 도입한 학습용 비율이다 [합성: S1,S2,S4]. 1번은 `import 성공 = deploy 성공`이라는 오해이고 [합성: S1,S2,S4], 3번은 `StableHLO 사용 = 자동 해결`이라는 오해이며 [합성: S1,S2,S3], 4번은 latency 지표와 lowering 준비도를 혼동한 경우다 [합성: S1,S2,S4].
- 객관식 2 정답: 1. StableHLO/XLA 계층은 높은 수준 연산 의미를 유지하는 쪽에 가깝지만, 그것이 곧 자동 최적 codegen 보장을 뜻하지는 않는다고 README가 설명했다 [S3][합성: S1,S2,S3]. 2번과 3번은 ONNX, StableHLO, MLIR을 같은 층으로 섞은 오해이고 [S1][S3][S4], 4번은 IREE의 compiler-runtime artifact path를 지워 버리는 오해다 [S2].
- 객관식 3 정답: 2. README의 디버깅 순서는 `입력 의미 -> MLIR legality -> IREE dispatch/artifact -> backend tuning`이다 [합성: S1,S2,S3,S4]. 1번은 backend tuning을 너무 앞에 둔 순서이고, 3번은 import 성공을 과신하는 순서이며, 4번은 학습 목표와 무관한 잡음이다 [합성: S1,S2,S3,S4].
- 짧은 서술형 1 예시: ONNX import 성공은 보통 모델 graph를 읽어 들였다는 뜻이지, 다음 단계가 요구하는 legality 조건을 모두 만족했다는 뜻은 아니다 [S4][합성: S1,S2,S4]. README의 `R_lowering`은 dialect legality, shape/type/layout, target lowering 준비 상태를 함께 보게 만드는 학습용 비율이므로, import만 통과한 상태에서는 아직 `R_lowering = 1`이라고 볼 수 없다 [합성: S1,S2,S4].
- 짧은 서술형 2 예시: decomposition은 backend-specific 최적화 이익 `B_backend`를 키울 수도 있지만, 동시에 dispatch 수를 늘려 `C_dispatch`와 경계 비용을 높일 수 있다 [합성: S1,S2,S3]. 따라서 README는 "더 잘게 내린다" 자체를 목표로 보지 않고, `G_net = B_backend - (C_dispatch + C_boundary + C_materialize)`가 실제로 양수인지로 판단하라고 제안했다 [합성: S1,S2,S3].
- 심화 설명형 1 예시: 먼저 이 문제가 ONNX graph 해석 문제인지, StableHLO 같은 높은 수준 의미 공간 문제인지 `입력 의미`부터 고정한다 [S3][S4]. 그다음 MLIR의 `dialect legality`를 보고 shape, type, rewrite, bufferization 중 어디가 다음 단계를 막는지 확인한다 [S1]. 이후 IREE의 `dispatch/artifact`를 확인해 compiler output이 runtime-loadable artifact로 이어지는 경로에서 무엇이 깨졌는지 본다 [S2]. 마지막에만 `backend tuning`을 본다. 앞의 세 단계가 정리되지 않았으면 backend 세부 조정은 진짜 원인을 가리지 못하기 때문이다 [합성: S1,S2,S3,S4].
- 심화 설명형 2 예시: 큰 연산 블록을 StableHLO 수준에서 조금 더 유지하는 선택은 높은 수준 의미를 남겨 뒤 단계에서 더 큰 최적화 여지를 줄 수 있지만, 이것이 `자동 최적화 보장 아님`이라는 점은 유지해야 한다 [S3][합성: S1,S2,S3]. 반대로 초기에 잘게 decomposition하면 어떤 target에서는 `B_backend`가 커질 수도 있지만, 동시에 `C_dispatch`와 `C_boundary`가 올라갈 위험이 있다 [합성: S1,S2,S3]. 그래서 README의 판단 기준은 어느 쪽 이름이 더 고급스러운가가 아니라, 실제로 `G_net`이 플러스인지 확인하는 것이다 [합성: S1,S2,S3].
