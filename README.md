### FineCo
---
This repository contains the source code for a research paper that was submitted for publication at the Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming 2024 (PPoPP 2024).

### What is FineCo
---
FineCo is a runtime system that can significantly improve the goodput of concurrent DNN inferences. The key idea of FineCo is to manage shared resources between kernels in inference requests in a fine-grained manner to alleviate resource competition and avoid resource waste.

### Environment Preparation
---
- Hardware&software requirements
  - Hardware Requirements
  - CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz
  - Memroy: 252G
  - NVIDIA Ampere 100 or AMD MI100

- Software Requirements
 - Ubuntu 20.04.1 (Kernel 5.8.0)
 - GPU Driver: 515.43.04
 - CUDA 11.7
 - Python 3.8.1

### Getting Started
---
The following sections step through the things required to run FineCo.
#### Code generation for fine-grained resource management （**Optional**）
- **Note**: For NVIDIA A100 and AMD MI100, we have placed the generated kernel in the FineCo runtime directory. For example, the implementation of the first depthwise Conv2d operator in MobileNet is in *ppopp24-fineco/runtime/fineco/include/kernel/cuda/ MobileNet/depthwise_conv2d_1*
- **Build FineCo Compiler**: If you want to generate multiple resource-aware codes on different devices, you need to compile our modified TVM：
````bash
cd compiler/tvm (CUDA backend) or cd compiler/tvm-hip (HIP backend)
mkdir build && cd build
cp ../config.cmake .
cmake .. && make -j 8
````
- **Generate resource-aware candidate code implementations**：
```bash
cd ppopp24-fineco/runtime/fineco/code_gen
python3 offline_run.py
```
#### Performance Evaluation
---
- Compiler performance evaluation
```bash
cd ppopp24-fineco/runtime/fineco/code_gen
python3 search_speace.py
```
- Runtime performance evaluation
```bash
bash build.sh
./main.out
```
