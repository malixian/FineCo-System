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
def conv_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 32, 3, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


@auto_scheduler.register_workload
def depthwise_conv2d_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 1, 32, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 64, 32, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, H, W), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 1, 64, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 128, 64, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 1, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 128, 128, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 1, 128, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 256, 128, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_5():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 1, 256, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_5():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 256, 256, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_6():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 1, 256, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_6():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 256, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_7():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 1, 512, 3, 3, (1, 1), (1,1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_7():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 512, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]
# dsc 7 * 5

@auto_scheduler.register_workload
def depthwise_conv2d_8():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 1, 512, 3, 3, (2, 2), (1, 1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_8():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 1024, 512, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def depthwise_conv2d_9():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 1, 1024, 3, 3, (1, 1), (1,1)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CI, CO, KH, KW), name="kernel")
    conv = topi.nn.depthwise_conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def pointwise_conv2d_9():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 1024, 1024, 1, 1, (1, 1), (0, 0)
    data = te.placeholder((N, CI, W, H), name="Input")
    # CO = channel_multiplier
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv)
    return [data, kernel, out]

@auto_scheduler.register_workload
def maxpool():
    N, C, H, W = 1, 1024, 7, 7
    kernel, stride, dilation, padding, pool_type = (7,7), (1, 1), (1,1), (0, 0, 0, 0), "avg"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def dense(batch=1, in_dim=1024, out_dim=1000):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]



if func_name == "conv_1":
    func = conv_1
elif func_name == "depthwise_conv2d_1":
    func = depthwise_conv2d_1
elif func_name == "pointwise_conv2d_1":
    func = pointwise_conv2d_1
elif func_name == "depthwise_conv2d_2":
    func = depthwise_conv2d_2
elif func_name == "pointwise_conv2d_2":
    func = pointwise_conv2d_2
elif func_name == "depthwise_conv2d_3":
    func = depthwise_conv2d_3
elif func_name == "pointwise_conv2d_3":
    func = pointwise_conv2d_3
elif func_name == "depthwise_conv2d_4":
    func = depthwise_conv2d_4
elif func_name == "pointwise_conv2d_4":
    func = pointwise_conv2d_4
elif func_name == "depthwise_conv2d_5":
    func = depthwise_conv2d_5
elif func_name == "pointwise_conv2d_5":
    func = pointwise_conv2d_5
elif func_name == "depthwise_conv2d_6":
    func = depthwise_conv2d_6
elif func_name == "pointwise_conv2d_6":
    func = pointwise_conv2d_6
elif func_name == "depthwise_conv2d_7":
    func = depthwise_conv2d_7
elif func_name == "pointwise_conv2d_7":
    func = pointwise_conv2d_7
elif func_name == "depthwise_conv2d_8":
    func = depthwise_conv2d_8
elif func_name == "pointwise_conv2d_8":
    func = pointwise_conv2d_8
elif func_name == "depthwise_conv2d_9":
    func = depthwise_conv2d_9
elif func_name == "pointwise_conv2d_9":
    func = pointwise_conv2d_9
elif func_name == "maxpool":
    func = maxpool
elif func_name == "dense":
    func = dense

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