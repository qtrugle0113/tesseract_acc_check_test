import os
import statistics
from acc import *
from acc_sklearn import *
from acc_js import *
from acc_sm import *
from acc_euc import *
from acc_man import *
#from text_recog import *

path = 'unseen_hw_data'

text_lists = list(file for file in os.listdir(path) if file.endswith(".gt.txt"))

text_lists = text_lists[:1]

vietrain_acc_list = []
viehw_acc_list = []

for text in text_lists:
    #acc_best, acc_train = get_accuracy(text, 'vie_best', 'vie_train')
    text_name = text.replace('.gt.txt', '.txt')
    acc_vietrain = sequence_matcher("unseen_vie_train_result/" + text_name, "unseen_hw_data/" + text)
    acc_viehw = sequence_matcher("unseen_vie_hw_result/" + text_name, "unseen_hw_data/" + text)

    vietrain_acc_list.append(acc_vietrain)
    viehw_acc_list.append(acc_viehw)

print(len(text_lists))
print(statistics.mean(vietrain_acc_list))
print(statistics.mean(viehw_acc_list))