# Rack-Scale Hardware and Networking Quiz

## 문제
1. disaggregated prefill/decode에서 KV cache transfer가 왜 별도 병목이 되는지 설명하라.
2. scale-up fabric과 scale-out network의 차이를 LLM serving 배치 예시로 설명하라.
3. topology-aware scheduling이 없을 때 같은 GPU 수의 요청이 왜 다른 latency를 보일 수 있는가?
4. power/cooling headroom을 serving capacity의 일부로 봐야 하는 이유를 설명하라.
5. MoE/expert parallel serving에서 rack boundary가 왜 중요해질 수 있는가?

## 정답 및 해설
1. prefill worker가 만든 KV cache를 decode worker가 읽어야 하므로, KV cache가 GPU 내부 상태가 아니라 네트워크와 memory ownership을 지나는 운영 자산이 된다. transfer가 늦으면 TTFT와 tail latency가 악화된다.
2. scale-up fabric은 한 rack이나 NVLink domain 안의 고대역폭/저지연 연결이고, scale-out network는 rack 사이 또는 cluster 전체 연결이다. TP/EP처럼 통신이 빽빽한 작업은 scale-up 안에 가깝게 두고, 독립 요청 병렬성은 scale-out으로 넓히기 쉽다.
3. scheduler가 rack, NVLink partition, network hop을 모르면 통신 밀도가 높은 작업을 멀리 떨어진 GPU에 배치할 수 있다. 이 경우 평균 GPU 사용률은 높아도 all-reduce, all-to-all, KV transfer 지연이 늘 수 있다.
4. reasoning이나 long-context workload는 지속 전력과 냉각 여유가 부족하면 throttling이나 service tier 하락으로 이어질 수 있다. 따라서 per-GPU 성능만이 아니라 facility budget 안에서 유지되는 useful tokens/sec를 봐야 한다.
5. MoE의 expert parallel은 token routing과 sparse all-to-all 통신을 만든다. expert shard가 rack boundary를 자주 넘으면 network hop과 tail latency가 커져 SLO가 흔들릴 수 있다.
