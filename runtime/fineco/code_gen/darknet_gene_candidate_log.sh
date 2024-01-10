cd darknet
export SET_TVM_BLOCK=16
python3 darknet_bench.py conv2d_19 16
export SET_TVM_BLOCK=20
python3 darknet_bench.py conv2d_19 20
export SET_TVM_BLOCK=25
python3 darknet_bench.py conv2d_19 25
export SET_TVM_BLOCK=28
python3 darknet_bench.py conv2d_19 28
export SET_TVM_BLOCK=35
python3 darknet_bench.py conv2d_19 35
export SET_TVM_BLOCK=40
python3 darknet_bench.py conv2d_19 40
export SET_TVM_BLOCK=49
python3 darknet_bench.py conv2d_19 49
export SET_TVM_BLOCK=50
python3 darknet_bench.py conv2d_19 50
export SET_TVM_BLOCK=56
python3 darknet_bench.py conv2d_19 56
export SET_TVM_BLOCK=70
python3 darknet_bench.py conv2d_19 70
export SET_TVM_BLOCK=80
python3 darknet_bench.py conv2d_19 80
export SET_TVM_BLOCK=98
python3 darknet_bench.py conv2d_19 98
export SET_TVM_BLOCK=100
python3 darknet_bench.py conv2d_19 100
