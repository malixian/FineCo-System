export SET_TVM_BLOCK=8
python3 bert_bench.py compute_qk 8
export SET_TVM_BLOCK=16
python3 bert_bench.py compute_qk 16
export SET_TVM_BLOCK=32
python3 bert_bench.py compute_qk 32
export SET_TVM_BLOCK=64
python3 bert_bench.py compute_sv 64
export SET_TVM_BLOCK=48
python3 bert_bench.py compute_sv 48
export SET_TVM_BLOCK=32
python3 bert_bench.py compute_sv 32
export SET_TVM_BLOCK=24
python3 bert_bench.py compute_sv 24
export SET_TVM_BLOCK=16
python3 bert_bench.py compute_sv 16
export SET_TVM_BLOCK=12
python3 bert_bench.py compute_sv 12
export SET_TVM_BLOCK=8
python3 bert_bench.py feed_forward_1 8
export SET_TVM_BLOCK=16
python3 bert_bench.py feed_forward_1 16
export SET_TVM_BLOCK=32
python3 bert_bench.py feed_forward_1 32
export SET_TVM_BLOCK=64
python3 bert_bench.py  feed_forward_1 64
export SET_TVM_BLOCK=19
python3 bert_bench.py feed_forward_2 19
export SET_TVM_BLOCK=57
python3 bert_bench.py  feed_forward_2 57
export SET_TVM_BLOCK=8
python3 bert_bench.py gene_qkv 8
export SET_TVM_BLOCK=16
python3 bert_bench.py gene_qkv 16
export SET_TVM_BLOCK=32
python3 bert_bench.py gene_qkv 32
