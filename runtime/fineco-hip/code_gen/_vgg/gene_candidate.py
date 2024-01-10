import os

if __name__ == '__main__':
    op = [
    "conv2d_block1_1", "conv2d_block1_2",
    "conv2d_block2_1", "conv2d_block2_2",
    "conv2d_block3_1", "conv2d_block3_2",
    "conv2d_block4_1", "conv2d_block4_2",
    "conv2d_block5_1",
    "maxpool_1", "maxpool_2", "maxpool_3", "maxpool_4", "maxpool_5",
    "dense_1", "dense_2", "dense_3"
    ]
    # cuNum = [98, 64, 56, 49, 32, 28, 16]
    cu = 100
    for p in op:
        # for cu in cuNum:
        #     os.environ["SET_TVM_BLOCK"] = str(cu)
        cmd = "python3 vgg_bench.py " + p + " " + str(cu)
        os.system(cmd)
            # os.system("echo $SET_TVM_BLOCK")
            # print(cmd)
        os.system("echo ---------------------------")