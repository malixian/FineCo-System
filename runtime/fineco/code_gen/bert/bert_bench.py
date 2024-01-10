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

batch_size = 1
embed_dim = 768 # equals hidden_dim 
head_num = 12
head_dim = 64
seq_len = 128
intermediate_size = 3072

'''
multi-head phase: gene_qkv ->  gene_w_qkv -> compute_qk -> compute_sv -> multi_head_concat
transformer_encoder: multi-head -> add_1 -> feed_forward_1 -> feed_forward_2 ->  add_2
'''

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

