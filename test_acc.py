import os
import statistics
from acc import *
from acc_sklearn import *
from acc_js import *
from acc_sm import *
from acc_euc import *
from acc_man import *
#from text_recog import *

path = 'ocr_data'

text_lists = list(file for file in os.listdir(path) if file.endswith(".gt.txt"))

#text_lists = text_lists[:]

best_acc_list = []
train_acc_list = []

for text in text_lists:
    #acc_best, acc_train = get_accuracy(text, 'vie_best', 'vie_train')
    acc_best = man_similarity("vie_best_result/" + text.replace('.gt', ''), "ocr_data/" + text)
    acc_train = man_similarity("vie_train_result/" + text.replace('.gt', ''), "ocr_data/" + text)

    best_acc_list.append(acc_best)
    train_acc_list.append(acc_train)

print(statistics.mean(best_acc_list))
print(statistics.mean(train_acc_list))