import matplotlib.pyplot as plt
# import sys
import os

path = '/home/adrian/Documents/dl2/final_results/'
dic1 = {}
dic2 = {}
files = []
# for i in range(len(sys.argv)):
filelist = os.listdir(path)
for i in sorted(filelist):
    if "small_12" in i:
    # if "_6" not in i:
        files.append(open(path+i, 'r'))
for inpfile in files:
    s = inpfile.name.split('/')
    fname = s[len(s)-1]
    dic1[fname] = []
    for line in inpfile:
        if 'epoch' in line:
            split = line.split(' Acc: ')
            print(line)
            acc = float(split[1])
            loss = split[0][15:22]
            # print(str(loss) + '    ' + acc)
            dic1[fname].append(acc)
        elif 'Eval' in line:
            evacc = line.split(' ')[3]
            print(evacc)
            evacc = float(evacc.strip())
            dic2[fname] = evacc


# Plot Training curves together
for logfile in dic1:
    plt.plot(dic1[logfile], label=logfile)

plt.legend(loc='best')
plt.ylabel("Acc")
plt.xlabel("Epoch")
plt.title("Comparison of Multitask DIRNets for classification")
plt.show()

## Barplot of eval accs:
# x = []
# y = []
# for key in dic2.keys():
#     x.append(key)
#     y.append(dic2[key])
# plt.bar(x,y)
# # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
# plt.title("Comparison of Multitask DIRNets for classification")
# plt.ylabel("Acc on Evaluation Set")
# plt.show()
