# Distributed Serving

한 장치의 kernel 최적화를 넘어 request phase, model shard, expert와 KV state를 여러 장치와 노드에 배치하는 방법을 다룬다.

## 챕터 순서
- [Disaggregated Prefill/Decode](./01-disaggregated-prefill-decode/README.md)
- [MoE Serving and Expert Parallelism](./02-moe-serving-and-expert-parallelism/README.md)
- [Scheduling and Admission Control](./03-scheduling-and-admission-control/README.md)
- [Model Parallelism: TP, PP, EP](./04-model-parallelism-tp-pp-ep/README.md)
- [Rack-Scale Hardware and Networking](./05-rack-scale-hardware-and-networking/README.md)
