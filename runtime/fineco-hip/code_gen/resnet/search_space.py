import subprocess

# kernel_name = "conv1"

def gen_search_space(kernel_name):
    file_name = "search_space_" + kernel_name + ".txt"

    wf = open(file_name, "w+")
    wf.write("grid  block   latency" + "\n")

    #ncu_cmd = "ncu -c 1 -k default_function_kernel0   --metrics launch__registers_per_thread,launch__shared_mem_per_block_allocated  python3 profile_by_id.py conv3 100"
    ncu_cmd = "python3 profile_by_id.py " + kernel_name + " 100"
    for i in range(0, 100):
        print("================ process ================", i)
        cmd = ncu_cmd + " " + str(i)
        #print("command is:", cmd)
        out = subprocess.getoutput(cmd)
        print(out)
        sub_outs = out.split("\n")
        has_grid_size = False
        write_line = ""
        for sub_out in sub_outs:
            if "grid size" in sub_out: 
                if has_grid_size:
                    continue
                has_grid_size = True
                rep_out = sub_out.replace("grid size:", "").replace("block size:", "").replace("\n", "")
                grid_block = rep_out.split(" ")
                grid = grid_block[0]
                block = grid_block[1]
                #print(grid, block)
                write_line += str(grid) + "  " +str(block)
            if "launch__registers_per_thread" in sub_out:
                register = sub_out.replace("launch__registers_per_thread", "").replace("register/thread", "").replace(" ", "").replace("\n", "")
                write_line += " " + str(register)
                #print(register)
            if "launch__shared_mem_per_block_allocated" in sub_out:
                shared_mem = sub_out.replace("launch__shared_mem_per_block_allocated", "").replace("Kbyte/block", "").replace(" ", "").replace("\n", "")
                write_line += " " + str(shared_mem)
            if "Execution time of this operator:" in sub_out:
                exec_time = sub_out.replace("Execution time of this operator:", "").replace("ms", "").replace(" ", "")
                write_line += " " + str(exec_time)
                print(exec_time)
        wf.write(write_line + "\n")
        if i % 20 == 0:
            wf.flush()
    wf.close()

def get_grid_perf():
    first_gflops = 456
    first_lat = 0.235
    file_name = "search_space_"+ kernel_name + ".txt"
    gen_file = "grid_perf_" + kernel_name + ".txt"
    rf = open(file_name, "r")
    wf = open(gen_file, "w")
    grid_best_perf = {}
    line_cnt =  0
    for line in rf.readlines():
        if line_cnt == 0:
            line_cnt += 1
            continue
        items = line.split(" ")
        grid = str(items[0])
        if int(items[0]) > 300:
            continue
        latency = str(items[3])
        print(items[3])
        perf = round(first_lat / float(items[3].replace(" ", "")) * first_gflops,2)
        if grid in grid_best_perf:
            if perf > grid_best_perf[grid]:
                grid_best_perf[grid] = perf
        else:
            grid_best_perf[grid] = perf
    for grid, perf in grid_best_perf.items():
        wline = grid + " " + str(perf) + "\n"
        wf.write(wline)

if __name__ == "__main__":
    for kernel_name in ["conv1", "conv2", "conv3"]:
        gen_search_space(kernel_name)
    #get_grid_perf()
