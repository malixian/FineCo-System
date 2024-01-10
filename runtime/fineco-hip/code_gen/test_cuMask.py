import os

if __name__ == '__main__':
    op = [["resnet", "conv1"], ["resnet", "conv2"], ["resnet", "conv3"],
            # ["vgg", "conv2d_block1_1"], ["vgg", "conv2d_block2_1"], ["vgg", "conv2d_block3_1"]
            ]
    # op = [["resnet", "conv1"]]
    cuMask = [[0xfffffffff, 0xfffffffff, 0xfffffffff, 0x3],      # 98
                [0xfffffffff, 0xfffffffff, 0x0, 0x0],      # 64
                [0xfffffffff, 0xfffffff00, 0x0, 0x0],      # 56
                [0xfefefefe, 0xfefefe00, 0x0, 0x0],             # 49
                [0xfffffffff, 0x0, 0x0, 0x0],             # 32
                [0xfefefefe, 0x0, 0x0, 0x0],              # 28
                [0xffff0000, 0x0, 0x0, 0x0]                     # 16
                ]
    cuNum = [98, 64, 56, 49, 32, 28, 16]
    # for p in op:
    #     cmd = "cd " + p[0] + " && python3 " + p[0] + "_test.py " + p[1] + " 100"
    #     for m in cuMask:
    #         os.environ["CUMASK0"] = hex(m[0])
    #         os.environ["CUMASK1"] = hex(m[1])
    #         os.environ["CUMASK2"] = hex(m[2])
    #         os.environ["CUMASK3"] = hex(m[3])
    #         os.system(cmd)
    #         os.system("echo ")
    #     os.system("echo ---------------------------")
    for p in op:
        for cu in cuNum:
            cmd = "cd " + p[0] + " && python3 " + p[0] + "_test.py " + p[1] + " " + str(cu)
            os.system(cmd)
            os.system("echo ")
        os.system("echo ---------------------------")