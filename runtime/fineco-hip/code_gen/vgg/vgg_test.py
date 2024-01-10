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
def conv2d_block1_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 64, 3, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_block1_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 64, 64, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


@auto_scheduler.register_workload
def maxpool2d_1():
    N, C, H, W = 1, 64, 224, 224
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def conv2d_block2_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 128, 64, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_block2_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 128, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def maxpool2d_2():
    N, C, H, W = 1, 128, 112, 112
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def conv2d_block3_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 256, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_block3_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 256, 256, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def maxpool2d_3():
    N, C, H, W = 1, 256, 56, 56
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def conv2d_block4_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 512, 256, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def conv2d_block4_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 512, 512, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]

@auto_scheduler.register_workload
def maxpool2d_4():
    N, C, H, W = 1, 512, 28, 28
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def conv2d_block5_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 512, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


@auto_scheduler.register_workload
def maxpool2d_5():
    N, C, H, W = 1, 512, 14, 14
    kernel, stride, dilation, padding, pool_type = (2,2), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def dense_1(batch=1, in_dim=25088, out_dim=4096):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]

@auto_scheduler.register_workload
def dense_2(batch=1, in_dim=4096, out_dim=4096):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]


@auto_scheduler.register_workload
def dense_3(batch=1, in_dim=4096, out_dim=1000):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]


if func_name == "conv2d_block1_1":
    func = conv2d_block1_1
elif func_name == "conv2d_block1_2":
    func = conv2d_block1_2
elif func_name == "conv2d_block2_1":
    func = conv2d_block2_1
elif func_name == "conv2d_block2_2":
    func = conv2d_block2_2
elif func_name == "conv2d_block3_1":
    func = conv2d_block3_1
elif func_name == "conv2d_block3_2":
    func = conv2d_block3_2
elif func_name == "conv2d_block4_1":
    func = conv2d_block4_1
elif func_name == "conv2d_block4_2":
    func = conv2d_block4_2
elif func_name == "conv2d_block5_1":
    func = conv2d_block5_1
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
elif func_name == "dense_1":
    func = dense_1
elif func_name == "dense_2":
    func = dense_2
elif func_name == "dense_3":
    func = dense_3


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
if func_name == "conv2d_block1_1":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 64, 3, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block1_2":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 64, 64, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block2_1":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 128, 64, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block2_2":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 112, 112, 128, 128, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block3_1":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 256, 128, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block3_2":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 56, 56, 256, 256, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block4_1":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 512, 256, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block4_2":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 28, 28, 512, 512, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "conv2d_block5_1":
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 14, 14, 512, 512, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,H,W)).astype(np.float32)

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

if func_name == "maxpool_1":
    N, C, H, W = 1, 64, 224, 224
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 112, 112)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool_2":
    N, C, H, W = 1, 128, 112, 112
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 56, 56)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool_3":
    N, C, H, W = 1, 256, 56, 56
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 28, 28)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool_4":
    N, C, H, W = 1, 512, 28, 28
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 14, 14)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool_5":
    N, C, H, W = 1, 512, 14, 14
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 7, 7)).astype(np.float32)

    dev = tvm.rocm()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )


elif func_name == "dense_1":
    Batch, In, Out = 1, 25088, 4096
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

elif func_name == "dense_2":
    Batch, In, Out = 1, 4096, 4096
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

elif func_name == "dense_3":
    Batch, In, Out = 1, 4096, 1000
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
# print(task.print_best(log_file, print_mode="rocm"))
