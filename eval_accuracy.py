import os
import statistics
from text_recog import *

path = 'unseen_hw_data'

img_lists = list(file for file in os.listdir(path) if file.endswith(".png"))

#img_lists = img_lists[:10]

best_acc_list = []
train_acc_list = []

for img in img_lists:
    acc_best, acc_train = get_accuracy(img, 'vie_train', 'vie_hw')
    best_acc_list.append(acc_best)
    train_acc_list.append(acc_train)

print(statistics.mean(best_acc_list))
print(statistics.mean(train_acc_list))
