import subprocess

file_name = "search_space_28.txt"

wf = open(file_name, "w+")
wf.write("grid  block   register/thread shared_mem/block" + "\n")

ncu_cmd = "ncu -c 1 -k default_function_kernel0   --metrics launch__registers_per_thread,launch__shared_mem_per_block_allocated  python3 profile_by_id.py conv1 28"
for i in range(0, 100):
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
            #print(shared_mem)
    wf.write(write_line + "\n")
    if i % 20 == 0:
        wf.flush()
wf.close()