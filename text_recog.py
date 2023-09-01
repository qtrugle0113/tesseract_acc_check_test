import pytesseract
from PIL import Image
from acc import *

def get_accuracy(img, lang1, lang2):
    path = img
    text_best = pytesseract.image_to_string(Image.open('unseen_hw_data/' + path), lang=lang1, config='--psm 13')

    text_train = pytesseract.image_to_string(Image.open('unseen_hw_data/' + path), lang=lang2, config='--psm 13')

    with open("unseen_vie_train_result/" + path.replace('png', 'txt'), "w", encoding="utf-8") as file:
        file.write(text_best)

    with open("unseen_vie_hw_result/" + path.replace('png', 'txt'), "w", encoding="utf-8") as file:
        file.write(text_train)

    acc_best = calculate_accuracy("unseen_vie_train_result/" + path.replace('png', 'txt'), "unseen_hw_data/" + path.replace('png', 'gt.txt'))
    acc_train = calculate_accuracy("unseen_vie_hw_result/" + path.replace('png', 'txt'), "unseen_hw_data/" + path.replace('png', 'gt.txt'))

    return acc_best, acc_train

# print(text_best)
# print(acc_best)
# print(text_train)
# print(acc_train)
