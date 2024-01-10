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

batch_size = 1
embed_dim = 768
head_num = 12
head_dim = 64 
seq_len = 128
intermediate_size = 3072 # 4 * embed_dim

@auto_scheduler.register_workload
def test_gemm():
    I = te.placeholder((1, 4, 4), name="I")
    W = te.placeholder((1, 4, 4), name="W")
    K = tvm.topi.nn.batch_matmul(I, W)
    return [I, W, K]

@auto_scheduler.register_workload
def gene_qkv():
    I = te.placeholder((batch_size, seq_len, embed_dim), name="I")
    W = te.placeholder((batch_size, head_dim, embed_dim), name="W")
    K = tvm.topi.nn.batch_matmul(I, W)
    return [I, W, K]

@auto_scheduler.register_workload
def compute_qk():
    Q = te.placeholder((batch_size, seq_len, head_dim), name="Q")
    K = te.placeholder((batch_size, seq_len, head_dim), name="K")
    out = tvm.topi.nn.batch_matmul(Q, K)
    return [Q, K, out]

@auto_scheduler.register_workload
def compute_sv():
    S = te.placeholder((batch_size, seq_len, seq_len), name="Q")
    V = te.placeholder((batch_size, embed_dim, seq_len), name="K")
    out = tvm.topi.nn.batch_matmul(S, V)
    return [S, V, out]

@auto_scheduler.register_workload
def feed_forward_1():
    I = te.placeholder((batch_size, seq_len, embed_dim), name="I")
    W = te.placeholder((batch_size, intermediate_size, embed_dim), name="W")
    out = tvm.topi.nn.batch_matmul(I, W)
    return [I, W, out]

@auto_scheduler.register_workload
def feed_forward_2():
    I = te.placeholder((batch_size, seq_len, intermediate_size), name="I")
    W = te.placeholder((batch_size, embed_dim, intermediate_size), name="W")
    out = tvm.topi.nn.batch_matmul(I, W)
    return [I, W, out]

if func_name == "gene_qkv":
    func = gene_qkv
elif func_name == "compute_qk":
    func = compute_qk
elif func_name == "compute_sv":
    func = compute_sv
elif func_name == "feed_forward_1":
    func = feed_forward_1
elif func_name == "feed_forward_2":
    func = feed_forward_2
elif func_name == "add":
    func = add
elif func_name == "test_gemm":
    func = test_gemm


task = auto_scheduler.SearchTask(
    func=func, args=(), target=target
)

sch, args = task.apply_best(log_file)
#sch, args = task.apply_by_id(log_file,log_id)

print("apply best config")

func = tvm.build(sch, args, target)

if func_name == "test_gemm" :
    i_np = np.random.uniform(size=(1, 4, 4)).astype(np.float32)
    w_np = np.random.uniform(size=(1, 4, 4)).astype(np.float32)
    dev = tvm.rocm()
    i_tvm = tvm.nd.array(i_np, device=dev)
    w_tvm = tvm.nd.array(w_np, device=dev)
    k_tvm = tvm.nd.empty((1, 4, 4), device=dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(i_tvm, w_tvm, k_tvm).results) * 1000)
    )


if func_name == "gene_qkv" :
    i_np = np.random.uniform(size=(batch_size, seq_len, embed_dim)).astype(np.float32)
    w_np = np.random.uniform(size=(batch_size, head_dim, embed_dim)).astype(np.float32)
    dev = tvm.rocm()
    i_tvm = tvm.nd.array(i_np, device=dev)
    w_tvm = tvm.nd.array(w_np, device=dev)
    k_tvm = tvm.nd.empty((batch_size, seq_len, head_dim), device=dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(i_tvm, w_tvm, k_tvm).results) * 1000)
    )


if func_name == "compute_qk":
    
    h1_np = np.random.uniform(size=(batch_size, seq_len, head_dim)).astype(np.float32)
    h2_np = np.random.uniform(size=(batch_size, seq_len,head_dim)).astype(np.float32)

    dev = tvm.rocm()
    h1_tvm = tvm.nd.array(h1_np, device=dev)
    h2_tvm = tvm.nd.array(h2_np, device=dev)
    o_tvm = tvm.nd.empty((batch_size, seq_len, seq_len), device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(h1_tvm, h2_tvm, o_tvm).results) * 1000)
    )

if func_name == "compute_sv":
    
    h1_np = np.random.uniform(size=(batch_size, seq_len, seq_len)).astype(np.float32)
    h2_np = np.random.uniform(size=(batch_size, embed_dim, seq_len)).astype(np.float32)

    dev = tvm.rocm()
    h1_tvm = tvm.nd.array(h1_np, device=dev)
    h2_tvm = tvm.nd.array(h2_np, device=dev)
    o_tvm = tvm.nd.empty((batch_size, seq_len, embed_dim), device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(h1_tvm, h2_tvm, o_tvm).results) * 1000)
    )

if func_name == "feed_forward_1":
    
    h1_np = np.random.uniform(size=(batch_size, seq_len, embed_dim)).astype(np.float32)
    h2_np = np.random.uniform(size=(batch_size, intermediate_size,embed_dim)).astype(np.float32)

    dev = tvm.rocm()
    h1_tvm = tvm.nd.array(h1_np, device=dev)
    h2_tvm = tvm.nd.array(h2_np, device=dev)
    o_tvm = tvm.nd.empty((batch_size, seq_len, intermediate_size), device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(h1_tvm, h2_tvm, o_tvm).results) * 1000)
    )


if func_name == "feed_forward_2":
    
    h1_np = np.random.uniform(size=(batch_size, seq_len, intermediate_size)).astype(np.float32)
    h2_np = np.random.uniform(size=(batch_size, embed_dim,intermediate_size)).astype(np.float32)

    dev = tvm.rocm()
    h1_tvm = tvm.nd.array(h1_np, device=dev)
    h2_tvm = tvm.nd.array(h2_np, device=dev)
    o_tvm = tvm.nd.empty((batch_size, seq_len, embed_dim), device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(h1_tvm, h2_tvm, o_tvm).results) * 1000)
    )

if func_name == "add":
    
    h1_np = np.random.uniform(size=(batch_size, seq_len, embed_dim)).astype(np.float32)
    h2_np = np.random.uniform(size=(batch_size, seq_len,embed_dim)).astype(np.float32)

    dev = tvm.rocm()
    h1_tvm = tvm.nd.array(h1_np, device=dev)
    h2_tvm = tvm.nd.array(h2_np, device=dev)
    o_tvm = tvm.nd.empty((batch_size, seq_len, embed_dim), device=dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(h1_tvm, h2_tvm, o_tvm).results) * 1000)
    )




#print("CUDA source code:")
#print(task.print_best(log_file, print_mode="cuda"))
