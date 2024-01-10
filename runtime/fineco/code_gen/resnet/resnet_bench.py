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
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 64, 3, 7, 7, (2, 2), (3, 3)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_3_s2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 128, 64, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]



@auto_scheduler.register_workload
def conv2d_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]



@auto_scheduler.register_workload
def conv2d_4_s2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 256, 128, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]



@auto_scheduler.register_workload
def conv2d_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


@auto_scheduler.register_workload
def conv2d_5_s2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 256, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]



@auto_scheduler.register_workload
def conv2d_5():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def maxpool2d():
    N, C, H, W = 1, 64, 112, 112
    kernel, stride, dilation, padding, pool_type = (3,3), (2, 2), (1,1), (1, 1, 1, 1), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def dense(batch=1, in_dim=25088, out_dim=1000):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]

@auto_scheduler.register_workload
def dense_1(batch=1, in_dim=100352, out_dim=1000):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]

@auto_scheduler.register_workload
def conv2d_2_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 64, 64, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


@auto_scheduler.register_workload
def conv2d_2_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_2_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 256, 64, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_2_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 64, 256, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


@auto_scheduler.register_workload
def conv2d_3_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 128, 256, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_3_2_ds():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 128, 128, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_3_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_3_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 512, 128, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_3_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 128, 512, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_4_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 256, 512, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_4_2_ds():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 256, 256, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_4_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_4_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 1024, 256, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_4_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 256, 1024, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


@auto_scheduler.register_workload
def conv2d_5_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 1024, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_5_2_ds():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 512, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_5_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_5_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 2048, 512, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_5_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 512, 2048, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

if func_name == "conv1":
    func = conv2d_1
elif func_name == "maxpool2d":
    func = maxpool2d
elif func_name == "conv2":
    func = conv2d_2
elif func_name == "conv3_s2":
    func = conv2d_3_s2
elif func_name == "conv3":
    func = conv2d_3
elif func_name == "conv4_s2":
    func = conv2d_4_s2
elif func_name == "conv4":
    func = conv2d_4
elif func_name == "conv5_s2":
    func = conv2d_5_s2
elif func_name == "conv5":
    func = conv2d_5
elif func_name == "dense":
    func = dense
elif func_name == "dense_1":
    func = dense_1
elif func_name == "conv2_1":
    func = conv2d_2_1
elif func_name == "conv2_2":
    func = conv2d_2_2
elif func_name == "conv2_3":
    func = conv2d_2_3
elif func_name == "conv2_4":
    func = conv2d_2_4
elif func_name == "conv3_1":
    func = conv2d_3_1
elif func_name == "conv3_2":
    func = conv2d_3_2
elif func_name == "conv3_3":
    func = conv2d_3_3
elif func_name == "conv3_4":
    func = conv2d_3_4
elif func_name == "conv3_2_ds":
    func = conv2d_3_2_ds
elif func_name == "conv4_1":
    func = conv2d_4_1
elif func_name == "conv4_2":
    func = conv2d_4_2
elif func_name == "conv4_3":
    func = conv2d_4_3
elif func_name == "conv4_4":
    func = conv2d_4_4
elif func_name == "conv4_2_ds":
    func = conv2d_4_2_ds
elif func_name == "conv5_1":
    func = conv2d_5_1
elif func_name == "conv5_2":
    func = conv2d_5_2
elif func_name == "conv5_3":
    func = conv2d_5_3
elif func_name == "conv5_4":
    func = conv2d_5_4
elif func_name == "conv5_2_ds":
    func = conv2d_5_2_ds


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
