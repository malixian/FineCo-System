export SET_TVM_BLOCK=64
python3 alexnet_bench.py dense_2 $SET_TVM_BLOCK

export SET_TVM_BLOCK=32
python3 alexnet_bench.py dense_2 $SET_TVM_BLOCK

export SET_TVM_BLOCK=16
python3 alexnet_bench.py dense_2 $SET_TVM_BLOCK

export SET_TVM_BLOCK=250
python3 alexnet_bench.py dense_3 $SET_TVM_BLOCK

export SET_TVM_BLOCK=125
python3 alexnet_bench.py dense_3 $SET_TVM_BLOCK

export SET_TVM_BLOCK=100
python3 alexnet_bench.py dense_3 $SET_TVM_BLOCK

export SET_TVM_BLOCK=50
python3 alexnet_bench.py dense_3 $SET_TVM_BLOCK

export SET_TVM_BLOCK=25
python3 alexnet_bench.py dense_3 $SET_TVM_BLOCK
