import os
import random
import re

random.seed(123)
data_folder = './dataset/'
data_re = re.compile(r"\d{6}\.csv")
files = [f for f in os.listdir(data_folder) if data_re.match(f)]
random.shuffle(files)
with open("files.txt", "w") as f:
    for file in files:
        with open(os.path.join(data_folder, file)) as csv:
            lines = csv.readlines()
            if len(lines) >= 243:
                f.write(file + "\n")