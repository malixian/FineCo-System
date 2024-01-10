import os

if __name__ == '__main__':
    op = [
        "conv2d_1","conv2d_2","conv2d_3","conv2d_4","conv2d_6",\
                    "conv2d_7",\
                    "conv2d_9",\
                    "conv2d_10",\
                    "conv2d_14",\
                    "conv2d_15",\
                    "conv2d_19",
                    "maxpool_1",\
                    "maxpool_2",\
                    "maxpool_3",\
                    "maxpool_4",\
                    "maxpool_5"
    ]
    # cuNum = [98, 64, 56, 49, 32, 28, 16]
    cu = 100
    for p in op:
        # for cu in cuNum:
        #     os.environ["SET_TVM_BLOCK"] = str(cu)
        cmd = "python3 darknet_bench.py " + p + " " + str(cu)
        os.system(cmd)
            # os.system("echo $SET_TVM_BLOCK")
            # print(cmd)
        os.system("echo ---------------------------")