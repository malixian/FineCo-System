import os, time, threading, sys
from multiprocessing.pool import ThreadPool
from multiprocessing import  Process
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
#import pdb

target = tvm.target.Target("rocm")
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

# Inspect the computational graph
#print("Computational DAG:")
#print(task.compute_dag)

# Apply the best schedule
sch, args = task.apply_best(log_file)

print("apply best config")

func = tvm.build(sch, args, target)

# Test Performance
if func_name == "conv_1":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 224, 224, 32, 3, 3, 3, (2, 2), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,112,112)).astype(np.float32)
    bias_np = np.random.uniform(size=(N,CO,1,1)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_1":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 1, 32, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,H,W)).astype(np.float32)
    

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    
    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_1":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 112, 112, 64, 32, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,112,112)).astype(np.float32)
    

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_2":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 1, 64, 3, 3, (2, 2), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,56,56)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_2":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 56, 56, 128, 64, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,56,56)).astype(np.float32)


    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )


if func_name == "depthwise_conv2d_3":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 1, 128, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,H,W)).astype(np.float32)


    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_3":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 56, 56, 128, 128, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,56,56)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_4":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 1, 128, 3, 3, (2, 2), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,28,28)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_4":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 28, 28, 256, 128, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,28,28)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_5":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 1, 256, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,H,W)).astype(np.float32)


    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_5":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 28, 28, 256, 256, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,28,28)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_6":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 1, 256, 3, 3, (2, 2), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,14,14)).astype(np.float32)


    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_6":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 512, 256, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,14,14)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )


if func_name == "depthwise_conv2d_7":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 1, 512, 3, 3, (1, 1), (1,1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,H,W)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_7":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 14, 14, 512, 512, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,14,14)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_8":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 1, 512, 3, 3, (2, 2), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,7,7)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_8":
    N, H, W, CO, CI, KH, KW, strides, padding =  1, 7, 7, 1024, 512, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,7,7)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "depthwise_conv2d_9":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 7, 7, 1, 1024, 3, 3, (1, 1), (1,1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CI, CO, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CI,H,W)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "pointwise_conv2d_9":
    N, H, W, CO, CI, KH, KW, strides, padding =  1, 7, 7, 1024, 1024, 1, 1, (1, 1), "SAME"
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,7,7)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool":
    N, C, H, W = 1, 1024, 7, 7
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 1, 1)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "dense":
    Batch, In, Out = 1, 1024, 1000
    data_np = np.random.uniform(size=(Batch, In)).astype(np.float32)
    weight_np = np.random.uniform(size=(Out, In)).astype(np.float32)
    bias_np = np.random.uniform(size=(Out,)).astype(np.float32)
    out_np = np.random.uniform(size=(1, Out)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    func(data_tvm, weight_tvm, bias_tvm, out_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )


# print("CUDA source code:")
# print(task.print_best(log_file, print_mode="cuda"))


