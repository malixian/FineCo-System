import os, math


def gene_candidates(model_name, kernel_name):
    grid_size, block_size, latency = get_block_latency(model_name, kernel_name, 100)
    grid_test_list = {}
    grid_factors = find_factor(grid_size)
    block_factors = find_factor(block_size)
    print(grid_factors)
    print(block_factors)
    all_factors = {}
    for factor in grid_factors:
        if factor in all_factors:
            continue
        else:
            all_factors[factor] = True
    
    for factor in block_factors:
        if factor in all_factors:
            continue
        else:
            all_factors[factor] = True
    tmp_factors_list = []
    for k in all_factors.keys():
        tmp_factors_list.append(k)
    tmp_factors_list.sort()

    all_factors_list = []
    for i in range(0, len(tmp_factors_list)):
        for j in range(i, len(tmp_factors_list)):
            all_factors_list.append(tmp_factors_list[i] * tmp_factors_list[j])

    tmp_candidate_factor_list = {}
    for factor in all_factors_list:
        if factor >= 15 and factor<=108 and factor not in tmp_candidate_factor_list:
            tmp_candidate_factor_list[factor] = True

    candidate_factor_list = []
    for factor in tmp_candidate_factor_list.keys():
        candidate_factor_list.append(factor)
    candidate_factor_list.sort()
    
    model_kernel_bench(model_name,kernel_name,candidate_factor_list)



def model_kernel_bench(model_name, kernel_name, test_list):
    f = open(model_name + "_gene_candidate_log.sh", "a+")
    #cmd = "cd " + model_name + "\n"
    #f.write(cmd)
    for val in test_list:
        print("run kernel %s, mps %d " % (kernel_name, val))
        cmd = "export SET_TVM_BLOCK=" + str(val) + "\n" 
        f.write(cmd)
        #os.system(cmd)
        cmd = "python3 " + model_name +"_bench.py " + kernel_name + " " + str(val) + "\n"
        #os.system(cmd)
        f.write(cmd)

def get_kernel_grid_size(model_name, kernel_name):
    cmd = "cd " + model_name + " && " +"ls | grep " + kernel_name + "-.*json"
    result = os.popen(cmd)  
    res = result.read()  
    grid_list = []
    for line in res.splitlines():
        line = line.replace(kernel_name, "").replace("-", "").replace(".json", "")
        grid_list.append(int(line))
    print(kernel_name)
    print(grid_list)
    return grid_list


def generate_resnet_candidate_kernel():
    kernels = [
        "conv1",
        "conv2","conv3_s2", "conv3",
        "conv4_s2", "conv4",
        "conv5_s2", "conv5",
        "conv2_1", "conv2_2", "conv2_3", "conv2_4",
        "conv3_1", "conv3_2", "conv3_2_ds", "conv3_3", "conv3_4",
        "conv4_1", "conv4_2", "conv4_2_ds",  "conv4_3", "conv4_4",
        "conv5_1", "conv5_2", "conv5_2_ds",  "conv5_3", "conv5_4"
    ]
    for k in kernels:
        gene_candidates("resnet", k) 



def gene_kernel_by_name_resnet(sub_model_name, layer_list):
    for layer_idx in range(1, len(layer_list)+1):
        layer = layer_list[layer_idx-1]
        for kernel_idx in range(1, len(layer)+1):
            kernel = layer[kernel_idx-1]
            kernel_name = ""
            if sub_model_name == "ResNet18a34":
                if kernel_idx == 1:
                    kernel_name = "conv1"
                else:
                    kernel_name = "conv_ds"
            elif sub_model_name == "ResNet50s":
                name_list_1 = ["conv1", "conv2", "conv3", "conv4"]
                name_list_2 = ["conv1", "conv2", "conv2_ds", "conv3", "conv4"]
                if layer_idx <= 2:
                    kernel_name = name_list_1[kernel_idx-1]
                else:
                    kernel_name = name_list_2[kernel_idx-1]
                
            gene_dir =  "../include/kernel/cuda/" + sub_model_name + "/" + "layer" + str(layer_idx) + "/" + kernel_name
            cmd = "mkdir -p " + gene_dir
            result = os.popen(gene_dir) 
            res = result.read()
            print(res)
            grid_size_list = get_kernel_grid_size("resnet", kernel)
            for i in range(0, len(grid_size_list)):
                btf = open(gene_dir + "/" + "candidate_params.txt", "a+")
                kf = open(gene_dir +  "/" + "candidate" + str(i) + ".cu", "w")
                cmd = "cd resnet && python3 resnet_test.py " + kernel + " " + str(grid_size_list[i])
                result = os.popen(cmd)  
                res = result.read()  
                line_cnt = 0
                for line in res.splitlines():
                    if line_cnt == 0:
                        bt = line.replace("grid size:", "").replace("block size:", "")
                        #btf.write("Block Thread" + "\n")
                        btf.write(bt+ "\n")
                    line_cnt += 1
                    if line_cnt > 15:
                        if "default_function_kernel0" in line:
                            line = line.replace("default_function_kernel0", "candidate" + str(i))
                        kf.write(line + "\n")
                kf.close()
                btf.close()

def applay_resnet_candidate_kernel(network):
    layer_list = []
    if network == "ResNet18a34":
        layer1 = ["conv1"]
        layer2 = ["conv2"]
        layer3 = ["conv3", "conv3_s2"]
        layer4 = ["conv4", "conv4_s2"]
        layer5 = ["conv5", "conv5_s2"]
        layer_list.append(layer1)
        layer_list.append(layer2)
        layer_list.append(layer3)
        layer_list.append(layer4)
        layer_list.append(layer5)
    elif network == "ResNet50s":
        layer1 = ["conv1"]
        layer2 = ["conv2_1", "conv2_2", "conv2_3", "conv2_4"]
        layer3 = ["conv3_1", "conv3_2", "conv3_2_ds", "conv3_3", "conv3_4"]
        layer4 = ["conv4_1", "conv4_2", "conv4_2_ds",  "conv4_3", "conv4_4"]
        layer5 = ["conv5_1", "conv5_2", "conv5_2_ds",  "conv5_3", "conv5_4"]
        layer_list.append(layer1)
        layer_list.append(layer2)
        layer_list.append(layer3)
        layer_list.append(layer4)
        layer_list.append(layer5)
    gene_kernel_by_name_resnet(network, layer_list)

def gene_applay_resnet_candidate(sub_model_name):
    generate_resnet_candidate_kernel()
    applay_resnet_candidate_kernel(sub_model_name)


def generate_candidate_kernel(sub_model_name):
    kernels = []
    if sub_model_name == "mobilenet":
        kernels = ["conv_1", "depthwise_conv2d_1","pointwise_conv2d_1", \
            "depthwise_conv2d_2", "pointwise_conv2d_2", "depthwise_conv2d_3", "pointwise_conv2d_3",\
            "depthwise_conv2d_4", "pointwise_conv2d_4", "depthwise_conv2d_5","pointwise_conv2d_5",\
            "depthwise_conv2d_6","pointwise_conv2d_6","depthwise_conv2d_7","pointwise_conv2d_7",\
            "depthwise_conv2d_8","pointwise_conv2d_8","depthwise_conv2d_9","pointwise_conv2d_9"
            #"maxpool","dense"
        ]
    elif sub_model_name == "vgg":
        kernels = ["conv2d_block1_1", "conv2d_block1_2", "conv2d_block2_1","conv2d_block2_2",\
            "conv2d_block3_1","conv2d_block3_2","conv2d_block4_1","conv2d_block4_2",\
            "conv2d_block5_1"
            #"maxpool_1","maxpool_2","maxpool_3","maxpool_4", "maxpool_5",
            #"dense_1","dense_2", "dense_3"
        ]
    elif sub_model_name == "darknet":
        kernels = [ "conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_6", "conv2d_7",\
            "conv2d_9","conv2d_10","conv2d_14","conv2d_15", "conv2d_19",\
            #"maxpool_1","maxpool_2","maxpool_3","maxpool_4","maxpool_5" 
            ]
    elif sub_model_name == "bert":
        kernels = [ "gene_qkv", "gene_w_qkv", "compute_qk",
            "compute_sv", "multi_head_concat_1", "multi_head_concat_2",
            "multi_head_concat_3", "multi_head_concat_4", "feed_forward_1", "feed_forward_2", "add"
        ]
    for k in kernels:
        gene_candidates(sub_model_name, k)


def gene_kernel_by_name(kernel_list, sub_model_name):
    model_mapping = {
        "vgg":"VGG",
        "mobilenet":"MobileNet",
        "bert":"Bert",
        "darknet" : "DarkNet"
    }
    model_file_name = model_mapping[sub_model_name]
    for kernel in kernel_list:
        gene_dir = "mkdir -p " + " ../include/kernel/cuda/" + model_file_name + "/" +  kernel + "/"
        result = os.popen(gene_dir) 
        res = result.read()
        grid_size_list = get_kernel_grid_size(sub_model_name, kernel)
        
        before_lat = 999
        first_add = True

        for i in range(0, len(grid_size_list)):
            btf = open("../include/kernel/cuda/" + model_file_name + "/" +  kernel + "/" + "candidate_params.txt", "a+")
            kf = open("../include/kernel/cuda/" + model_file_name + "/" +  kernel +  "/" + "candidate" + str(i) + ".cu", "w")
            print("../include/kernel/cuda/" + model_file_name + "/" +  kernel + "/" + "candidate_params.txt")
            grid_size = grid_size_list[i]
            cmd = "cd " + sub_model_name + " && python3 " + sub_model_name + "_test.py " + kernel + " " + str(grid_size)
            
            result = os.popen(cmd)  
            res = result.read()  
            line_cnt = 0
            bt = ""
            for line in res.splitlines():
                if "Execution time of this operator" in line:
                    lat = line.replace("Execution time of this operator:", "").replace("ms", "").\
                    replace(" ", "").replace("\n","")
                    print(bt, line)
                    lat = float(lat)
                    if first_add:
                        first_add = False
                        btf.write(bt + "\n")
                        btf.write(str(lat) + "\n")
                    elif before_lat > lat :
                        before_lat = lat
                        print("change before_lat:", before_lat)
                        btf.write(bt + "\n")
                        btf.write(str(lat) + "\n")
                    else:
                        break
                    #btf.write(line + "\n")
                if line_cnt == 0:
                    bt = line.replace("grid size:", "").replace("block size:", "")
                    #btf.write("Block Thread" + "\n")
                
                line_cnt += 1
                if line_cnt > 15:
                    if "default_function_kernel0" in line:
                        line = line.replace("default_function_kernel0", "candidate" + str(i))
                    kf.write(line + "\n")
            kf.close()
            btf.close()

def applay_candidate_kernel(sub_model_name):
    kernels = []
    if sub_model_name == "mobilenet":
        kernels = ["conv_1", "depthwise_conv2d_1","pointwise_conv2d_1", \
            "depthwise_conv2d_2", "pointwise_conv2d_2", "depthwise_conv2d_3", "pointwise_conv2d_3",\
            "depthwise_conv2d_4", "pointwise_conv2d_4", "depthwise_conv2d_5","pointwise_conv2d_5",\
            "depthwise_conv2d_6","pointwise_conv2d_6","depthwise_conv2d_7","pointwise_conv2d_7",\
            "depthwise_conv2d_8","pointwise_conv2d_8","depthwise_conv2d_9","pointwise_conv2d_9",\
            #"maxpool","dense"
        ]
    elif sub_model_name == "vgg":
        kernels = ["conv2d_block1_1", "conv2d_block1_2", "conv2d_block2_1","conv2d_block2_2",\
            "conv2d_block3_1","conv2d_block3_2","conv2d_block4_1","conv2d_block4_2",\
            "conv2d_block5_1",
            #"maxpool_1","maxpool_2","maxpool_3","maxpool_4",\
            #"maxpool_5","dense_1","dense_2", "dense_3"
        ]
    elif sub_model_name == "darknet":
        kernels = [ "conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_6",\
            "conv2d_7","conv2d_9","conv2d_10","conv2d_14","conv2d_15",\
            "conv2d_19",
            #"maxpool_1","maxpool_2","maxpool_3","maxpool_4","maxpool_5" ]
        ]
    elif sub_model_name == "bert":
        kernels = [ "gene_qkv", "gene_w_qkv", "compute_qk",
            "compute_sv", "multi_head_concat_1", "multi_head_concat_2",
            "multi_head_concat_3", "multi_head_concat_4", "feed_forward_1", "feed_forward_2", "add"
        ]
    gene_kernel_by_name(kernels, sub_model_name)

def gene_applay_dnn_candidate(sub_model_name):
    #generate_candidate_kernel(sub_model_name)
    applay_candidate_kernel(sub_model_name)






def generate_kernel(layer_list, sub_model_name, layer_mps):
    layer_idx = 1
    layer_idx_str = "layer" + str(layer_idx)
    has_conv = True
    for kernels in layer_list:
        for task_idx in range(0, len(kernels)):
            if "conv" in kernels[task_idx]:
                has_conv = True
                conv_name_str = kernels[task_idx]
                gene_dir = "mkdir -p " + " ../include/kernel/cuda/" + sub_model_name + "/" +  layer_idx_str + "/" +  conv_name_str + "/"
                result = os.popen(gene_dir) 
                res = result.read() 
                print(gene_dir)    
                for i in range(0, len(layer_mps[conv_name_str])):
                    btf = open("../include/kernel/cuda/" + sub_model_name + "/" +  layer_idx_str + "/" +  conv_name_str + "/" + "candidate_params.txt", "a+")
                    kf = open("../include/kernel/cuda/" + sub_model_name + "/" +  layer_idx_str + "/" +  conv_name_str + "/" + "candidate" + str(i) + ".cu", "w")
                    mps = layer_mps[conv_name_str]
                    cmd = "cd resnet && python3 resnet_test.py " + kernels[task_idx] + " " + str(mps[i])
                    result = os.popen(cmd)  
                    res = result.read()  
                    line_cnt = 0
                    for line in res.splitlines():
                        if line_cnt == 0:
                            bt = line.replace("grid size:", "").replace("block size:", "")
                            btf.write("Block Thread")
                            btf.write(bt + "\n")
                        line_cnt += 1
                        if line_cnt > 6:  
                            kf.write(line + "\n")
                    kf.close()
                    btf.close()
            else:
                kernel_str = kernels[task_idx]
                gene_dir = "mkdir -p " + " ../include/kernel/cuda/" + sub_model_name + "/" +  kernel_str + "/"
                result = os.popen(gene_dir)
                has_conv = False
                for i in range(0, len(layer_mps[kernel_str])):
                    btf = open("../include/kernel/cuda/" + sub_model_name + "/" + kernel_str + "/" + "candidate_params.txt", "a+")
                    kf = open("../include/kernel/cuda/" + sub_model_name + "/" + kernel_str + "/" + "candidate" + str(i) + ".cu", "w")
                    mps = layer_mps[kernel_str]
                    cmd = "cd resnet && python3 resnet_test.py " + kernels[task_idx] + " " + str(mps[i])
                    result = os.popen(cmd)  
                    res = result.read()  
                    line_cnt = 0
                    for line in res.splitlines():
                        if line_cnt == 0:
                            bt = line.replace("grid size:", "").replace("block size:", "")
                            btf.write("Block Thread")
                            btf.write(bt + "\n")
                        line_cnt += 1
                        if line_cnt > 6:  
                            kf.write(line + "\n")
                    kf.close()
                    btf.close()
            #print("write " + layer_idx_str + " " + conv_idx_str + " successfully")
        if has_conv:
            layer_idx += 1
            layer_idx_str = "layer" + str(layer_idx)



def find_factor(n):
    if n == 0: return {0}
    if n == 1: return {1}
 
    rlist = {1}
    i = 2
    while i <= n:
        if n % i == 0:
            rlist.add(i)
        i += 1
    print(sorted(rlist))
    return sorted(rlist)


def get_block_latency(model_name, kernel_name, mps):
    cmd = "cd " + model_name + " && python3 " + model_name + "_test.py " + kernel_name + " " + str(mps)
    result = os.popen(cmd)  
    res = result.read()  
    line = res.splitlines()[0]
    print("res line:", line)
    grid_block_size = line.replace("grid size:", "").replace("block size:", "").split(" ")
    grid_size = int(grid_block_size[0])
    block_size = int(grid_block_size[1])

    line = res.splitlines()[len(res.splitlines())-1]
    latency = line.replace("Execution time of this operator:", "").replace("ms", "").replace(" ", "")
    print("block_size %s, latency %s" % (block_size, latency))
    
    return grid_size, block_size, latency

'''
def gene_candidates(model_name, kernel_name):
    block_size, latency = get_block_latency(model_name, kernel_name, 100)

    mps_test_list = []
    if block_size <= 108:
        factors = find_factor(block_size)
        # currently default generate 3 candidates
        # currently default min mps is 20, there should set layer qos
        for factor in factors:
            used_sm = block_size / factor
            mps = math.ceil(used_sm / 108 * 100)
            if len(mps_test_list)<3 and mps >= 20 and mps <= 75:
                mps_test_list.append(mps)
    else:
        blocks_per_sm = math.ceil(block_size / 108)
        left_sm = blocks_per_sm * 108 - block_size
        full_sm = 108 - left_sm
        compressed_sm = left_sm * (blocks_per_sm-1) / blocks_per_sm
        used_sm = full_sm + compressed_sm
        factors = find_factor(used_sm)
        for factor in factors:
            used_sm = used_sm / factor
            mps = math.ceil(used_sm / 108 * 100)
            if len(mps_test_list)<3 and mps >= 20 and mps <= 75:
                mps_test_list.append(mps)


    for mps in mps_test_list:
        model_kernel_bench(model_name,kernel_name,mps)
'''


def gene_vgg_candidate():
    kernels= ["conv2d_block1_1", "conv2d_block1_2", \
    #"maxpool_1","maxpool_2","maxpool_3","maxpool_4","maxpool_5",
    "conv2d_block2_1", "conv2d_block2_2", \
     "conv2d_block3_1",\
    #"conv2d_block3_2",  "conv2d_block4_1", "conv2d_block4_2", \
    #"conv2d_block5_1",
    #"dense_1","dense_2","dense_3"
    ]
    for kernel in kernels:
        gene_candidates("vgg", kernel)

def gene_mobilenet_candidate():
    kernels = ["conv_1", \
        "depthwise_conv2d_1",\
        "pointwise_conv2d_1",\
        "depthwise_conv2d_2",\
        "pointwise_conv2d_2",\
        "depthwise_conv2d_3",\
        "pointwise_conv2d_3",\
        "depthwise_conv2d_4",\
        "pointwise_conv2d_4",\
        "depthwise_conv2d_5",\
        "pointwise_conv2d_5",\
        "depthwise_conv2d_6",\
        "pointwise_conv2d_6",\
        "depthwise_conv2d_7",\
        "pointwise_conv2d_7",\
        "depthwise_conv2d_8",\
        "pointwise_conv2d_8",\
        "depthwise_conv2d_9",\
        "pointwise_conv2d_9",
        #"maxpool",\
        #"dense"
    ]
    for kernel in kernels:
        gene_candidates("mobilenet", kernel)


def gene_darknet_candidate():
    kernels = ["conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_6",\
                    "conv2d_7",\
                    "conv2d_9",\
                    "conv2d_10",\
                    "conv2d_14",\
                    "conv2d_15",\
                    "conv2d_19",
                    #"maxpool_1",\
                    #"maxpool_2",\
                    #"maxpool_3",\
                    #"maxpool_4",\
                    #"maxpool_5"]
    ]
    for kernel in kernels:
        gene_candidates("darknet", kernel)

def gene_bert_candidate():
    kernels = [
        "compute_qk",
        "compute_sv",
        "feed_forward_1",
        "feed_forward_2",
        "gene_qkv"
    ]
    for kernel in kernels:
        gene_candidates("bert", kernel)



if __name__ == "__main__":
    # Example: generate candidate kernel by tvm and apply for fienco
    # ***************** auto scheduling kernel ******************
    # *** Step 1 generate candidate kernel ***
    generate_resnet_candidate_kernel()
    # *** Step 2 apply candidate kernel ***
    applay_resnet_candidate_kernel("ResNet50s")
    gene_applay_dnn_candidate("darknet")
    gene_applay_dnn_candidate("vgg")
    gene_applay_dnn_candidate("mobilenet")


    
    
    
