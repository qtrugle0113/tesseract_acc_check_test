import os
import re
import statistics

path = 'test_data'

img_lists = [file for file in os.listdir(path) if re.search(r'\.(png|jpg|jpeg|tif)$', file, re.I)]
label_lists = [file for file in os.listdir(path) if re.search(r'\.(txt|gt.txt)$', file, re.I)]

print(len(img_lists))
print(len(label_lists))
