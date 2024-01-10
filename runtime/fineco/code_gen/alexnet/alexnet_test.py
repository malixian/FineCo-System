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
def conv2d_relu_layer_1():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 224, 224, 48, 3, 11, 11, (4, 4), (2, 2)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_relu_layer_2():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 27, 27, 128, 48, 5, 5, (1, 1), (2, 2)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_relu_layer_3():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 13, 13, 192, 128, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]

@auto_scheduler.register_workload
def conv2d_relu_layer_4():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 13, 13, 192, 192, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]



@auto_scheduler.register_workload
def conv2d_relu_layer_5():
    N, H, W, CO, CI, KH, KW, stride, padding = 1, 13, 13, 128, 192, 3, 3, (1, 1), (1, 1)
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]
@auto_scheduler.register_workload
def maxpool2d_1():
    N, C, H, W = 1, 48, 55, 55
    kernel, stride, dilation, padding, pool_type = (3,3), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def maxpool2d_2():
    N, C, H, W = 1, 128, 27, 27
    kernel, stride, dilation, padding, pool_type = (3,3), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def maxpool2d_3():
    N, C, H, W = 1, 128, 13, 13
    kernel, stride, dilation, padding, pool_type = (3,3), (2, 2), (1,1), (0, 0, 0, 0), "max"
    data = te.placeholder((N, C, H, W), name="data")
    pool2d = tvm.topi.nn.pool2d(data, kernel, stride, dilation, padding, pool_type)
    return [data, pool2d]

@auto_scheduler.register_workload
def dense_1(batch=1, in_dim=4608, out_dim=2048):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]



@auto_scheduler.register_workload
def dense_2(batch=1, in_dim=2048, out_dim=2048):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]


@auto_scheduler.register_workload
def dense_3(batch=1, in_dim=2048, out_dim=500):
    data = te.placeholder((batch, in_dim), name="data")
    weight = te.placeholder((out_dim, in_dim), name="weight")
    bias = te.placeholder((out_dim, ), name="bias")
    dense = topi.nn.dense(data, weight, bias)
    return [data, weight, bias, dense]

if func_name == "conv1":
    func = conv2d_relu_layer_1
elif func_name == "conv2":
    func = conv2d_relu_layer_2
elif func_name == "conv3":
    func = conv2d_relu_layer_3
elif func_name == "conv4":
    func = conv2d_relu_layer_4
elif func_name == "conv5":
    func = conv2d_relu_layer_5
elif func_name == "maxpool2d_1":
    func = maxpool2d_1
elif func_name == "maxpool2d_2":
    func = maxpool2d_2
elif func_name == "maxpool2d_3":
    func = maxpool2d_3
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


#task.filter_valid(log_file)
#task.generate_code_base(log_file, "valid_idx_alex_layer1.txt", func_name)


# Apply the best schedule
sch, args = task.apply_best(log_file)

#print("Lowered TIR:")
#print(tvm.lower(sch, args, simple_mode=True))



func = tvm.build(sch, args, target)

# Test Performance
if func_name == "conv1":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 224, 224, 48, 3, 11, 11, (4, 4), (2, 2)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,55,55)).astype(np.float32)
    bias_np = np.random.uniform(size=(1,CO,1,1)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )


if func_name == "conv2":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 27, 27, 128, 48, 5, 5, (1, 1), (2, 2)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,27,27)).astype(np.float32)
    bias_np = np.random.uniform(size=(1,CO,1,1)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )

if func_name == "conv3":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 13, 13, 192, 128, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,13,13)).astype(np.float32)
    bias_np = np.random.uniform(size=(1,CO,1,1)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )

if func_name == "conv4":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 13, 13, 192, 192, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,13,13)).astype(np.float32)
    bias_np = np.random.uniform(size=(1,CO,1,1)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )



if func_name == "conv5":
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 13, 13, 128, 192, 3, 3, (1, 1), (1, 1)
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    conv_np = np.random.uniform(size=(N,CO,13,13)).astype(np.float32)
    bias_np = np.random.uniform(size=(1,CO,1,1)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    out_tvm = tvm.nd.empty(conv_np.shape, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool2d_1":
    N, C, H, W, KH, KW = 1, 48, 55, 55, 3, 3
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 27, 27)).astype(np.float32)    

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)
    
    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool2d_2":
    N, C, H, W, KH, KW = 1, 128, 27, 27, 3, 3
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 13, 13)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )

if func_name == "maxpool2d_3":
    N, C, H, W, KH, KW = 1, 128, 13, 13, 3, 3
    data_np = np.random.uniform(size=(N, C, H, W)).astype(np.float32)
    out_np = np.random.uniform(size=(N, C, 6, 6)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, out_tvm).results) * 1000)
    )



elif func_name == "dense_1":
    Batch, In, Out = 1, 4608, 2048
    data_np = np.random.uniform(size=(Batch, In)).astype(np.float32)
    weight_np = np.random.uniform(size=(Out, In)).astype(np.float32)
    bias_np = np.random.uniform(size=(Out,)).astype(np.float32)
    out_np = np.random.uniform(size=(1, Out)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)

    func(data_tvm, weight_tvm, bias_tvm, out_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )
elif func_name == "dense_2":
    Batch, In, Out = 1, 2048, 2048
    data_np = np.random.uniform(size=(Batch, In)).astype(np.float32)
    weight_np = np.random.uniform(size=(Out, In)).astype(np.float32)
    bias_np = np.random.uniform(size=(Out,)).astype(np.float32)
    out_np = np.random.uniform(size=(1, Out)).astype(np.float32)

    dev = tvm.cuda()
    data_tvm = tvm.nd.array(data_np, device=dev)
    weight_tvm = tvm.nd.array(weight_np, device=dev)
    bias_tvm = tvm.nd.array(bias_np, device=dev)
    out_tvm = tvm.nd.array(out_np, device=dev)
    
    func(data_tvm, weight_tvm, bias_tvm, out_tvm)
    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
    )

elif func_name == "dense_3":
    Batch, In, Out = 1, 2048, 500
    data_np = np.random.uniform(size=(Batch, In)).astype(np.float32)
    weight_np = np.random.uniform(size=(Out,In)).astype(np.float32)
    bias_np = np.random.uniform(size=(Out,)).astype(np.float32)
    out_np = np.random.uniform(size=(1, Out)).astype(np.float32)

    dev = tvm.cuda()
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

#print("CUDA source code:")
#print(task.print_best(log_file, print_mode="cuda"))

