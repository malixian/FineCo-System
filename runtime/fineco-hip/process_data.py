# [cu_num][layer] -> [layer][cu_num]
def tranverse(cunum):
    m = len(cunum)
    n = len(cunum[0])
    for j in range(n):
        layer = []
        for i in range(m):
            layer.append(cunum[i][j])
        print(layer)

if __name__ == '__main__':
    cunum = []
    layer = []
    cumask = 0
    model_name = ""
    with open("test_kernel_cu_num.log") as f:
        for line in f:
            if line[0] == '@':
                if model_name != "":
                    cunum.append(layer)
                    print("\n" + model_name + " " + str(cumask))
                    tranverse(cunum)
                    # break
                line = f.readline()
                begin = line.find('_') + 1
                end = line.find('\n')
                model_name = line[begin:end]
                line = f.readline()
            elif line[:4] == "pack":
                cumask = 0
            elif line[:4] == "dist" or line[:4] == "cons":
                cunum.append(layer)
                print("\n" + model_name + " " + str(cumask))
                tranverse(cunum)
                cumask = cumask + 1
            elif line[0] == '\n':
                cunum = []
                layer = []
            elif line.find("Model Init Successfully") != -1:
                if layer != []:
                    cunum.append(layer)
                    layer = []
            elif line[0] == '=':
                continue
            else:
                begin = line.find("Avg") + 13
                layer.append(float(line[begin:]))
    print("\n" + model_name + " " + str(cumask))
    tranverse(cunum)

