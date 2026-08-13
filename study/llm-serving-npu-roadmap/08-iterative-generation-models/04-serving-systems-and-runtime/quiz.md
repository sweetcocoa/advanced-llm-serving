# 반복 생성 모델의 Serving Systems and Runtime 퀴즈

## 객관식 1
2026-08-12 기준 vLLM의 DiffusionGemma 지원에 대한 설명으로 가장 정확한 것은 무엇인가?

1. 별도 Python loop가 vLLM API 앞에서 동작할 뿐 scheduler와는 통합되지 않는다.
2. Model Runner V2의 `ModelState`와 diffusion sampler를 사용하고 기존 speculative decoding data path를 재사용하는 native 지원이다.
3. 모든 masked diffusion LLM이 같은 코드 경로에서 자동 지원된다.
4. conventional AR KV cache와 완전히 같은 계산량을 보장한다.

## 객관식 2
Ouro model card가 안내하는 vLLM inference의 핵심 제한은 무엇인가?

1. tensor parallel을 전혀 사용할 수 없다.
2. chat template을 사용할 수 없다.
3. adaptive exit를 지원하지 않아 깊이 적응의 계산 절감을 살리지 못한다.
4. Ouro weight를 로드하지 못한다.

## 객관식 3
Fast-dLLM, dKV-Cache, Sparse-dLLM의 차이를 가장 정확하게 설명한 것은 무엇인가?

1. 세 방법 모두 causal exact KV cache를 그대로 적용한다.
2. Fast-dLLM은 prefix reuse와 parallel decoding, dKV-Cache는 token 상태에 따른 delayed caching, Sparse-dLLM은 saliency 기반 bidirectional cache eviction을 사용한다.
3. 세 방법 모두 vLLM mainline release에 포함돼 있다.
4. 세 방법의 최고 speedup은 hardware와 workload에 무관하다.

## 객관식 4
dInfer에 대한 설명으로 가장 적절한 것은 무엇인가?

1. vLLM의 다른 이름이다.
2. model, diffusion iteration manager, decoder, KV-cache manager로 나눈 공개 dLLM inference framework이며 일부 backend에 vLLM/SGLang의 특정 버전을 사용한다.
3. DiffusionGemma만 지원하는 official Google engine이다.
4. 연구 코드가 없고 paper proposal만 존재한다.

## 객관식 5
Sangam이 deficit token budget을 다음 scheduling round로 이월하는 주된 이유는 무엇인가?

1. 모든 prefill을 token 하나씩 chunking하기 위해서다.
2. in-flight decode를 보호하면서 indivisible refresh prefill이 영원히 굶지 않게 하기 위해서다.
3. model weight를 host memory로 이동하기 위해서다.
4. rollback된 token을 다시 출력하기 위해서다.

## 객관식 6
Archer의 cache boundary에 대한 설명으로 가장 정확한 것은 무엇인가?

1. mutable response K/V만 영구 cache하고 prompt는 매번 계산한다.
2. prompt와 response를 모두 무기한 cache한다.
3. mutable response는 매 step 재계산하고 prompt K/V만 bounded reuse한 뒤 state 변화에 따라 refresh한다.
4. KV cache를 전혀 사용하지 않는다.

## 객관식 7
Continuous depth batching이 일반 token-level continuous batching과 다른 이유는 무엇인가?

1. prompt tokenization을 GPU에서 하기 때문이다.
2. 서로 다른 token을 recurrent loop iteration 사이에서 exit/refill하고 boundary stage와 loop stage를 다른 빈도로 schedule해야 하기 때문이다.
3. 모든 token이 항상 같은 loop 수를 사용하기 때문이다.
4. diffusion canvas만 처리하기 때문이다.

## 짧은 서술형 1
`vLLM이 Ouro를 실행한다`와 `vLLM이 Ouro의 adaptive-depth serving을 지원한다`가 왜 다른 주장인지 3~4문장으로 설명하라. 답변에는 `fixed depth`, `adaptive exit`, `scheduler`를 포함하라.

## 짧은 서술형 2
bidirectional diffusion decoding에서 conventional AR KV cache가 exact하지 않은 이유를 설명하라. 답변에는 `response token`, `K/V`, `refresh`를 포함하라.

## 짧은 서술형 3
dInfer의 저자 보고 1,100 TPS를 자신의 production capacity 수치로 바로 사용할 수 없는 이유를 3가지 적어라.

## 심화 설명형 1
LLaDA 기반 online service에서 cache refresh가 길어 진행 중 decode의 p99를 악화시키고 있다. Fast-dLLM/dKV-Cache류의 cache와 Sangam scheduler를 함께 고려해 진단 및 실험 순서를 제안하라. 답변에는 `refresh interval`, `indivisible prefill`, `deficit budget`, `quality baseline`을 포함하라.

## 심화 설명형 2
rollback 가능한 diffusion code generator를 가속하려 한다. response-side aggressive cache와 Archer의 prompt-only bounded reuse를 비교하고, 어떤 지표를 측정할지 설명하라. 답변에는 `staleness`, `Pass@1`, `latency`, `rollback`을 포함하라.

## 심화 설명형 3
Ouro와 Huginn을 같은 adaptive-depth endpoint에서 운영한다고 가정하라. CDB가 boundary queue와 loop queue를 분리하는 이유, Ouro와 Huginn의 구조 차이가 refill 이득에 미치는 영향을 설명하라.

## 판정 문제
다음 문장에 각각 `릴리스된 엔진 지원`, `제한된 엔진 실행`, `공개 연구 구현`, `논문 구현/제안` 중 하나를 붙이고 근거를 한 문장으로 적어라.

- DiffusionGemma via vLLM MRV2
- Ouro via vLLM
- dInfer v0.2
- Fast-dLLM
- Archer
- Continuous depth batching for Ouro/Huginn

## 정답 및 해설
- 객관식 1 정답: 2. DiffusionGemma는 MRV2 `ModelState`로 canvas와 per-request state를 관리하고 diffusion-specific sampler를 사용하면서 기존 speculative decoding path를 재사용한다. native 지원은 모든 dLLM의 자동 지원이나 AR과 동일한 cache 비용을 뜻하지 않는다. [S1][S2][S3]
- 객관식 2 정답: 3. Ouro model card는 vLLM 경로에서 adaptive exit가 지원되지 않는다고 명시한다. 모델은 fixed depth로 실행할 수 있지만 token 난이도별 compute 절감은 얻지 못한다. [S5]
- 객관식 3 정답: 2. 세 방법은 안정성을 가정하는 대상과 cache 관리가 다르다. 모두 bidirectional state에 근사를 도입하므로 동일 workload의 quality validation이 필요하다. [S7][S9][S10]
- 객관식 4 정답: 2. dInfer는 dLLM-specific component를 분리한 공개 framework다. 일부 실행에서 pinned vLLM/SGLang backend를 사용하지만 upstream vLLM native feature와 동일하지 않다. [S11][S12]
- 객관식 5 정답: 2. dLLM refresh prefill을 임의로 chunking하기 어렵기 때문에 budget이 찰 때까지 이월한다. 진행 중 decode를 우선하면서도 prefill starvation을 막는 장치다. [S13]
- 객관식 6 정답: 3. Archer는 언제 바뀔지 모르는 response 전체를 current하게 유지하고, identity가 고정된 prompt state만 제한된 기간 재사용한다. generation이 anchor에서 멀어지면 prompt도 refresh한다. [S14]
- 객관식 7 정답: 2. depth-adaptive model은 한 token의 forward 내부에서도 exit가 발생한다. CDB는 loop iteration을 scheduling 단위로 만들고 prelude/coda 같은 boundary stage를 별도 queue에서 처리한다. [S16]
- 짧은 서술형 1 예시: vLLM은 Ouro를 fixed depth로 실행할 수 있지만 현재 adaptive exit를 살리는 scheduler를 제공하지 않는다. adaptive exit는 token이 서로 다른 recurrent depth에서 나가야 한다. 기존 scheduler가 full forward 경계에서만 batch를 바꾸면 먼저 끝난 token의 계산 slot을 즉시 회수할 수 없다. [S5][S16]
- 짧은 서술형 2 예시: bidirectional attention에서는 response token 하나가 바뀌면 다른 response와 prompt representation의 K/V도 영향을 받는다. AR처럼 과거 K/V가 영구히 exact하다고 볼 수 없으므로 cache 기반 방법은 일정한 refresh와 staleness 관리가 필요하다. [S7][S9][S14]
- 짧은 서술형 3 예시: 1,100 TPS는 LLaDA-MoE, batch 1 HumanEval, 8x H800이라는 저자 조건의 결과다. production traffic의 arrival distribution, streaming/cancellation, p99 latency를 측정하지 않는다. speed-only script와 품질 평가는 분리돼 있으므로 동일 output quality도 별도 확인해야 한다. [S11][S12]
- 심화 설명형 1 예시: 먼저 eager full-refresh를 quality baseline으로 고정하고 refresh interval별 accuracy와 latency를 측정한다. 그다음 trace에서 indivisible prefill 크기와 decode stall을 확인한다. Sangam식 deficit budget으로 in-flight decode를 먼저 처리하되 미사용 budget을 이월해 refresh prefill starvation을 막고, colocated와 hybrid를 같은 arrival trace에서 비교한다. [S9][S13]
- 심화 설명형 2 예시: response-side aggressive cache는 rollback으로 바뀐 token에서 만든 stale K/V를 남길 수 있다. Archer는 response를 매번 재계산하고 prompt만 bounded reuse해 rollback 자유를 유지한다. cache radius별 Pass@1, compile/test success, p50/p99 latency와 rollback 횟수를 paired baseline과 비교해야 한다. [S14]
- 심화 설명형 3 예시: Ouro는 fully looped 구조라 boundary 비용이 작고 빈 slot refill의 이득을 회수하기 쉽다. Huginn은 prelude-recurrent-coda 구조여서 coda와 loop를 같은 빈도로 실행하면 낭비가 생긴다. CDB는 boundary와 loop priority queue를 분리해 각 stage를 충분한 batch로 묶고, 한 step 앞선 exit decision으로 loop slot을 refill한다. [S16][S17]
- 판정 문제 예시: DiffusionGemma는 릴리스된 엔진 지원이다. Ouro via vLLM은 adaptive exit가 빠진 제한된 엔진 실행이다. dInfer, Fast-dLLM, Archer는 코드가 공개된 연구 구현이다. CDB는 논문에서 end-to-end 평가됐지만 2026-08-12 기준 범용 engine release로 확인되지 않은 논문 구현/제안이다. [S1][S5][S8][S11][S14][S16]
