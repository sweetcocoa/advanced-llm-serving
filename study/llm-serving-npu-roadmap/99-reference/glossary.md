# Glossary

| 용어 | 설명 |
| --- | --- |
| Prefill | prompt 전체를 한 번에 읽어 KV cache를 만드는 단계 |
| Decode | 새 토큰을 한 개씩 생성하는 반복 단계 |
| KV Cache | attention 계산 재사용을 위해 저장하는 key/value 상태 |
| TTFT | Time To First Token, 첫 토큰까지 걸리는 시간 |
| TPOT | Time Per Output Token, 토큰당 평균 생성 시간 |
| Continuous Batching | 요청이 도중 합류/이탈할 수 있는 동적 batching 방식 |
| Execution Provider | ONNX Runtime에서 특정 backend로 graph 일부를 보내는 단위 |
| Offload Coverage | 전체 graph 중 accelerator 위에서 실행되는 비율 |
| Structured Outputs | schema 제약을 만족하는 형식으로 토큰을 생성하는 방식 |
| Hybrid Execution | CPU/GPU/NPU가 한 workload를 나눠 처리하는 실행 방식 |
| Context Binary | 특정 backend나 device에 맞게 미리 컴파일된 실행 산출물 |
| Core ML | Apple의 on-device ML 프레임워크 및 model packaging/runtime 생태계 |
| QNN | Qualcomm Neural Network SDK 및 backend 생태계를 가리키는 표현 |
| OGA | On-device Generative AI, AMD Ryzen AI 문맥에서 hybrid 실행 워크플로를 설명할 때 자주 쓰이는 표현 |
| TPU | Tensor Processing Unit, Google의 전용 AI accelerator 계열 |
