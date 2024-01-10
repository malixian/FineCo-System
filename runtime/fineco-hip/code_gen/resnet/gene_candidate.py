import os

if __name__ == '__main__':
    op = [
        # "conv1",
        # "conv2","conv3_s2", "conv3",
        "conv3",
        # "conv4_s2", "conv4",
        # "conv5_s2", "conv5",
        # "conv2_1", "conv2_2", "conv2_3", "conv2_4",
        # "conv3_1", "conv3_2", "conv3_2_ds", "conv3_3", "conv3_4",
        # "conv4_1", "conv4_2", "conv4_2_ds",  "conv4_3", "conv4_4",
        # "conv5_1", "conv5_2", "conv5_2_ds",  "conv5_3", "conv5_4"
    ]
    cuNum = [98, 64, 56, 49, 32, 28, 16]
    cu = 100
    for p in op:
        # for cu in cuNum:
        #     os.environ["SET_TVM_BLOCK"] = str(cu)
        cmd = "python3 resnet_bench.py " + p + " " + str(cu)
        os.system(cmd)
            # os.system("echo $SET_TVM_BLOCK")
            # print(cmd)
        os.system("echo ---------------------------")