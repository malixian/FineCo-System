def gen_figure(file):
    wf = open("resnet50-result.txt", "w")
    start = 0
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.replace("size: ","").replace("effective latency: ", "").replace("\n", "").replace("layer idx:", "")
            items = line.split(" ", )
            size = int(items[1])
            latency = int(float(items[2]))
            for i in range(0, latency):
                start += 1
                wline = str(start) + " " + str(size) + "\n"
                wf.write(wline)
    wf.close()

# Simple Moving Average, SMA
def gen_sma_figure(file):
    wf = open("resnet101-sma-result.txt", "w")
    start = 0
    windows = 200
    sm_num = 0
    try_size = 0
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            items = line.split(" ", )
            size = int(items[1])
            try_size += 1
            if try_size <= windows:
                sm_num += size
            else:
                print(sm_num)
                avg_sm_num = int(sm_num / windows)
                try_size = 0
                sm_num = 0
                for ts in range(0, windows):
                    wline = str(start) + " " + str(avg_sm_num) + "\n"
                    wf.write(wline)
                    start += 1
    wf.close()


# Simple Moving Mode, SMM
def gen_smm_figure(file):
    wf = open("resnet101-smm-result.txt", "w")
    start = 0
    windows = 500
    try_size = 0
    sm_occur = {}
    with open(file, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            items = line.split(" ", )
            size = int(items[1])
            try_size += 1
            if try_size <= windows:
                if size in sm_occur.keys():
                    sm_occur[size] += 1 + int(size/20)
                else :
                    sm_occur[size] = 1
            else:
                sm_occur_list = sorted(sm_occur.items(), key=lambda x: x[1], reverse=True)
                try_size = 0
                sm_num = 0
                print(sm_occur_list[0])
                mode_number, repeat = sm_occur_list[0]
                sm_occur = {}
                for ts in range(0, windows):
                    wline = str(start) + " " + str(mode_number) + "\n"
                    wf.write(wline)
                    start += 1
    wf.close()


def compute_avg_util(file):
    max_util = 0
    avg_util = 0
    with open(file, "r") as f:
        for line in f.readlines():
            items = line.split(" ")
            sm = int(items[1])
            max_util += 108
            avg_util += sm
    print("util ratio is:", avg_util/max_util)

if __name__ == "__main__":
    file_path = "resnet101-config.txt"
    #gen_figure(file_path)
    #gen_smm_figure("resnet101-result.txt")
    compute_avg_util(file_path)

