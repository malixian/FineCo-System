import os

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

if __name__ == '__main__':
    model_name, kernel_name = "resnet", "conv1"
    grid_size, block_size, latency = get_block_latency(model_name, kernel_name, 100)
    print(grid_size, block_size, latency)