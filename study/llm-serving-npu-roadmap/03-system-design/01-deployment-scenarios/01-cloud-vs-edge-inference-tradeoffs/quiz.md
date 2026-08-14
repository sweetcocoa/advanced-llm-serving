# Cloud vs Edge Inference Tradeoffs 퀴즈

## 객관식 1
고객사 회의 내용을 영업사원 노트북에서 요약해야 한다. 원문 외부 전송은 최소화해야 하고, 현장 네트워크도 불안정하다. 가장 먼저 검토할 기본 배치로 가장 적절한 것은 무엇인가?
1. pure cloud
2. edge 또는 local-first hybrid
3. 브라우저 UI 변경
4. 모델 리더보드 재정렬

## 객관식 2
긴 정책 문서를 읽고 답해야 하는 글로벌 고객센터 assistant에서 클라우드가 유리한 이유로 가장 적절한 것은 무엇인가?
1. 디바이스 발열이 없어서
2. 긴 문맥과 높은 동시성을 중앙에서 더 유연하게 처리할 수 있어서
3. 프라이버시 요구가 자동으로 사라져서
4. 네트워크 비용이 항상 0이라서

## 객관식 3
edge 실험에서 "NPU 지원"이라고 표시되지만 실제 체감 속도가 기대보다 나쁘다. 가장 먼저 추가로 확인해야 할 항목은 무엇인가?
1. 마케팅 슬로건
2. offload coverage와 CPU fallback 비율
3. 발표 영상 조회 수
4. 모델 카드 색상

## 짧은 서술형 1
`private inference`가 왜 곧바로 `pure edge`와 같은 뜻이 아닌지 2~3문장으로 설명하라.

## 짧은 서술형 2
클라우드 TTFT 식

$$
T_{\mathrm{TTFT}}^{\mathrm{cloud}} = T_{\mathrm{uplink}} + T_{\mathrm{queue}} + T_{\mathrm{prefill}} + T_{\mathrm{first\ decode}}
$$

을 보고, 사용자가 "오늘 첫 응답이 유난히 늦다"고 말했을 때 어떤 순서로 병목을 의심할지 적어라.

## 심화 설명형 1
다음 두 서비스 중 하나를 골라 적절한 placement를 설계하라.
- 사내 회의 요약 assistant
- 글로벌 고객센터 답변 초안 assistant

답변에는 `latency`, `privacy`, `cost`, `power` 네 축 중 무엇을 우선하는지와, 왜 cloud/edge/hybrid를 택하는지 포함하라.

## 심화 설명형 2
Windows PC 기반 on-device assistant를 설계한다고 가정하자. Windows ML, QNN EP, OpenVINO NPU, Ryzen AI hybrid workflow 같은 문서들이 왜 "edge가 가능하다"는 주장만이 아니라 "실행 경로를 어떻게 쪼갤 것인가"라는 질문으로 읽혀야 하는지 설명하라.

## 정답 및 해설
- 객관식 1 정답: 2. 네트워크 불안정성과 원문 최소 전송 요구가 있으므로 edge 또는 local-first hybrid가 출발점이다.
- 객관식 2 정답: 2. 긴 문맥과 높은 동시성은 중앙 자원 조정과 클라우드 최적화의 이점을 크게 만든다. disaggregated prefill과 speculative decoding 같은 서버 측 전략도 여기에 연결된다.
- 객관식 3 정답: 2. "지원" 여부만으로는 충분하지 않다. 실제로 얼마나 NPU에 올라가는지와 CPU fallback이 얼마나 자주 생기는지가 체감 성능을 좌우한다.
- 짧은 서술형 1 예시: private inference는 원문 데이터나 민감 정보를 불필요하게 외부로 내보내지 않는 설계 목표를 뜻한다. 따라서 로컬 마스킹이나 분류 후 필요한 정보만 cloud로 보내는 hybrid도 private inference 요구를 만족할 수 있다.
- 짧은 서술형 2 예시: 먼저 회선 문제와 업로드 지연을 본다. 그다음 피크 시간 큐 적체를 보고, 이후 긴 프롬프트 때문에 prefill이 길어진 것인지 확인한다. 마지막으로 decode 최적화 부족인지 점검한다.
- 심화 설명형 1 해설 포인트: 회의 요약은 privacy와 즉시 반응성을 우선해 edge 또는 hybrid가 자연스럽다. 고객센터 초안은 긴 문맥, 동시성, 중앙 업데이트 요구 때문에 cloud가 유리하다.
- 심화 설명형 2 해설 포인트: 각 문서는 단순한 "지원 목록"이 아니라 어느 연산을 어느 런타임과 장치로 보낼지에 대한 힌트를 준다. 따라서 좋은 답안은 offload coverage, fallback, 전처리/후처리 분리, selective escalation 같은 용어를 포함해야 한다.
