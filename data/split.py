import os
import random
import re

random.seed(123)
data_folder = './dataset/'
sgzt_file = './sgzt.csv'
sgzt_file_tmp = './sgzt.csv.tmp'
data_re = re.compile(r"\d{6}\.csv")
files = [f for f in os.listdir(data_folder) if data_re.match(f)]
random.shuffle(files)
sgzt = {}


def load_sgzt():
    if not os.path.exists(sgzt_file):
        if os.path.exists(sgzt_file_tmp):
            os.rename(sgzt_file_tmp, sgzt_file)
        else:
            f = open(sgzt_file, 'w')
            f.close()

    with open(sgzt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            fund_id, state = line.strip().split(",")
            if state == "True" or state == "False":
                sgzt[f'{fund_id}.csv'] = state
            else:
                raise ValueError(f"state {state} not supported yet")
load_sgzt()
with open("trainval_files.txt", "w") as trainval_f:
    with open("test_files.txt", "w") as test_f:
        for file in files:
            with open(os.path.join(data_folder, file)) as csv:
                lines = csv.readlines()
                if len(lines) >= 243:
                    trainval_f.write(file + "\n")
                    if file in sgzt and sgzt[file]=='True':
                        test_f.write(file + "\n")