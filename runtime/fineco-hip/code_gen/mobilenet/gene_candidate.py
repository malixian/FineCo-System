import os

if __name__ == '__main__':
    op = [
        "conv_1", "depthwise_conv2d_1","pointwise_conv2d_1", \
        "depthwise_conv2d_2", "pointwise_conv2d_2", "depthwise_conv2d_3", "pointwise_conv2d_3",\
        "depthwise_conv2d_4", "pointwise_conv2d_4", "depthwise_conv2d_5","pointwise_conv2d_5",\
        "depthwise_conv2d_6","pointwise_conv2d_6","depthwise_conv2d_7","pointwise_conv2d_7",\
        "depthwise_conv2d_8","pointwise_conv2d_8","depthwise_conv2d_9","pointwise_conv2d_9",\
        "maxpool","dense"
    ]
    # cuNum = [98, 64, 56, 49, 32, 28, 16]
    cu = 100
    for p in op:
        # for cu in cuNum:
        #     os.environ["SET_TVM_BLOCK"] = str(cu)
        cmd = "python3 mobilenet_bench.py " + p + " " + str(cu)
        os.system(cmd)
            # os.system("echo $SET_TVM_BLOCK")
            # print(cmd)
        os.system("echo ---------------------------")