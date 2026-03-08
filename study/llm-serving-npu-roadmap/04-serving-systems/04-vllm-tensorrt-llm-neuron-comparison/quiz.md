# vLLM, TensorRT-LLM, Neuron Comparison 퀴즈

## 객관식 1
다음 중 이 챕터의 설명에 가장 잘 맞는 연결은 무엇인가?

1. vLLM은 release notes 검토가 우선이고, Neuron은 범용 serving 실험이 우선이다.
2. TensorRT-LLM은 prefill/decode 분리 검토와 가장 직접적으로 연결된다.
3. AWS Neuron은 장비 종속성이 거의 없어서 하드웨어 검토가 중요하지 않다.
4. 세 backend는 운영 질문이 같아서 같은 문서 순서로 평가하면 된다.

## 객관식 2
모델 교체가 잦고 빠르게 serving 실험을 시작해야 하는 팀이 있다. 이 챕터의 비교 틀에서 가장 자연스러운 출발점은 무엇인가?

1. vLLM을 기준선으로 두고 실험 속도와 serving 표면을 먼저 본다.
2. TensorRT-LLM만 고정하고 handoff 비용은 나중에 생각한다.
3. AWS Neuron announcement 없이 release notes도 건너뛴다.
4. 어떤 backend든 모델만 같으면 운영 비용도 같다고 본다.

## 객관식 3
AWS Neuron을 평가할 때 benchmark 전에 먼저 던져야 할 질문으로 가장 적절한 것은 무엇인가?

1. 현재 모델, SDK, 하드웨어 조합이 지원 범위 안에 있는가?
2. TensorRT-LLM의 disaggregated serving을 바로 적용할 수 있는가?
3. vLLM 문서만 읽어도 Neuron 운영 정책을 대신할 수 있는가?
4. 장비 종속성은 무시해도 되는가?

## 짧은 서술형 1
이 챕터에서 backend 비교를 "성능표"보다 "실패 비용표"에서 시작해야 한다고 한 이유를 2~3문장으로 설명하라.

## 짧은 서술형 2
`TTFT_split = T_queue + T_prefill + T_handoff + T_first_decode` 식이 TensorRT-LLM 검토에서 어떤 판단을 돕는지 설명하라.

## 심화 설명형 1
다음 세 팀 가운데 하나를 골라 어떤 backend를 1순위로 검토할지 쓰고, 그 이유를 생산성, 성능 상한, 장비 종속성, 검증 방식 네 축으로 설명하라.

- 팀 A: 모델 후보가 자주 바뀌고 serving 실험을 빨리 시작해야 함
- 팀 B: 긴 입력 비중이 크고 stage별 병목을 따로 다루고 싶음
- 팀 C: Inferentia/Trainium 기반 운영 표준화를 추진 중임

## 심화 설명형 2
vLLM에서 출발한 서비스가 성장하면서 TensorRT-LLM 또는 AWS Neuron 전환을 검토한다. 단순 처리량 외에 반드시 비교해야 할 항목을 네 가지 이상 제시하고, 왜 그 항목이 실제 운영 의사결정에 중요한지 설명하라.

## 정답 및 해설
- 객관식 1 정답: 2. TensorRT-LLM의 핵심 비교 포인트는 disaggregated serving, 즉 prefill/decode 분리 검토와 직접 연결된다. [S2]
- 객관식 2 정답: 1. 이 챕터에서는 vLLM을 빠른 serving 실험과 모델 교체가 필요한 팀의 기준선으로 둔다. [S1]
- 객관식 3 정답: 1. AWS Neuron은 announcement와 release notes를 함께 보며 현재 모델, SDK, 하드웨어 조합의 지원 여부를 먼저 확인해야 한다. [S3] [S4]
- 짧은 서술형 1 예시: backend 선택은 평균 성능 숫자 하나로 끝나지 않는다. 어떤 팀은 모델 교체 지연이 가장 비싸고, 어떤 팀은 stage 병목이나 장비 종속성이 더 비싸기 때문에 실패 비용을 먼저 정의해야 실제로 맞는 backend를 고를 수 있다.
- 짧은 서술형 2 예시: 이 식은 분리형 서빙에서 첫 토큰 시간의 구성 요소를 나눠 보게 만든다. 따라서 TensorRT-LLM이 주는 이득이 prefill/decode 분리에서 오는지, 아니면 handoff 비용 때문에 상쇄되는지를 판단하는 데 쓰인다. [S2]
- 심화 설명형 1 해설 포인트: 팀 A는 vLLM [S1], 팀 B는 TensorRT-LLM [S2], 팀 C는 AWS Neuron [S3] [S4]이 각각 1순위가 되기 쉽다. 정답의 핵심은 모델 이름보다 팀의 운영 방식과 검증 절차가 backend 우선순위를 만든다는 점을 설명하는 것이다.
- 심화 설명형 2 해설 포인트: 새 모델 온보딩 시간, stage 분리로 인한 handoff 비용, 운영자가 먼저 읽어야 할 문서, 장비 종속성, 릴리스 검증 절차, 지원 조합 확인 비용 등을 비교해야 한다. 이 챕터는 backend가 성능 숫자뿐 아니라 운영 문법과 장애 분석 순서를 바꾼다는 점을 강조한다. [S1] [S2] [S3] [S4]
