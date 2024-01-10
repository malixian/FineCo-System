import os, time, threading, sys
from multiprocessing.pool import ThreadPool
from multiprocessing import  Process
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
#import pdb

target = tvm.target.Target("cuda")
MPS = 100
func_name = ""
if len(sys.argv) < 3:
    print("Need Kernel Name and MPS Configuration")
    sys.exit()
else:
    func_name = str(sys.argv[1])
    MPS = str(sys.argv[2])

log_file = func_name + "-" + str(MPS) + ".json"
print(log_file)


@auto_scheduler.register_workload
def conv2d_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 32, 3, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 64, 32, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 128, 64, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 64, 128, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

# conv2d_5 equals conv2d_3

@auto_scheduler.register_workload
def conv2d_6():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 256, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


@auto_scheduler.register_workload
def conv2d_7():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 128, 256, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

# conv2d_8 equals conv2d_6

@auto_scheduler.register_workload
def conv2d_9():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 256, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


@auto_scheduler.register_workload
def conv2d_10():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 256, 512, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

# conv2d_11, conv2d_13 equals conv2d_9, conv2d_12 equals conv2d_10, 

@auto_scheduler.register_workload
def conv2d_14():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 1024, 512, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


@auto_scheduler.register_workload
def conv2d_15():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 512, 1024, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

# conv2d_16, conv2d_18 equals conv2d_14, conv2d_17 equals conv2d_15, 

@auto_scheduler.register_workload
def conv2d_19():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 1000, 1024, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def maxpool2d_1():
    N, C, H, W = 1, 32, 224, 224
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def maxpool2d_2():
    N, C, H, W = 1, 64, 112, 112
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def maxpool2d_3():
    N, C, H, W = 1, 128, 56, 56
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def maxpool2d_4():
    N, C, H, W = 1, 256, 28, 28
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def maxpool2d_5():
    N, C, H, W = 1, 512, 14, 14
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]



if func_name == "conv2d_1":
    func = conv2d_1
elif func_name == "conv2d_2":
    func = conv2d_2
elif func_name == "conv2d_3":
    func = conv2d_3
elif func_name == "conv2d_4":
    func = conv2d_4
elif func_name == "conv2d_6":
    func = conv2d_6
elif func_name == "conv2d_7":
    func = conv2d_7
elif func_name == "conv2d_9":
    func = conv2d_9
elif func_name == "conv2d_10":
    func = conv2d_10
elif func_name == "conv2d_14":
    func = conv2d_14
elif func_name == "conv2d_15":
    func = conv2d_15
elif func_name == "conv2d_19":
    func = conv2d_19
elif func_name == "maxpool_1":
    func = maxpool2d_1
elif func_name == "maxpool_2":
    func = maxpool2d_2
elif func_name == "maxpool_3":
    func = maxpool2d_3
elif func_name == "maxpool_4":
    func = maxpool2d_4
elif func_name == "maxpool_5":
    func = maxpool2d_5


task = auto_scheduler.SearchTask(
    func=func, args=(), target=target
)

cost_model = auto_scheduler.XGBModel()
cost_model.update_from_file(log_file)
search_policy = auto_scheduler.SketchPolicy(
    task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
)

print("init auto scheduler")
measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=100,  # change this to 1000 to achieve the best performance
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

# Run auto-tuning (search)
task.tune(tune_option, search_policy=search_policy)

# Kill the measurement process
del measure_ctx
