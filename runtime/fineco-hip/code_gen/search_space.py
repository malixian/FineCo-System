import subprocess

def gen_search_space():
    file_name = "search_space_bert.txt"

    wf = open(file_name, "w+")
    wf.write("grid  block perf  register/thread shared_mem/block" + "\n")

    ncu_cmd = "cd bert && ncu -c 1 -k default_function_kernel0   --metrics launch__registers_per_thread,launch__shared_mem_per_block_allocated  python3 bert_test.py test_gemm 100"
    for i in range(0, 500):
        print("================ process ================", i)
        cmd = ncu_cmd + " " + str(i)
        #print("command is:", cmd)
        out = subprocess.getoutput(cmd)
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
    first_gflops = 5756
    first_lat = 0.082
    file_name = "search_space.txt"
    gen_file = "grid_perf.txt"
    rf = open(file_name, "r")
    wf = open(gen_file, "w")
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
        wline = grid + " " + str(perf) + "\n"
        wf.write(wline)

if __name__ == "__main__":
    #get_grid_perf()
    gen_search_space()