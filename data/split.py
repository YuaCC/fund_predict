import os
import re
import random
random.seed(123)
data_folder = './dataset/'
data_re = re.compile(r"\d{6}\.csv")
files = [f for f in os.listdir(data_folder) if data_re.match(f)]
random.shuffle(files)
num = len(files)
files1,files2 = files[:num//2],files[num//2:]
with open("files1.txt","w") as f:
    for file in files1:
        with open(os.path.join(data_folder,file)) as csv:
            lines = csv.readlines()
            if len(lines)>=162:
                f.write(file+"\n")
with open("files2.txt","w") as f:
    for file in files2:
        with open(os.path.join(data_folder,file)) as csv:
            lines = csv.readlines()
            if len(lines)>=162:
                f.write(file+"\n")


