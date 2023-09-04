# 필요한 패키지 가져 오기
import os
import re
import pytesseract
import argparse
import statistics
import numpy as np
from PIL import Image
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from fuzzywuzzy import fuzz


# 텍스트 유사도 계산: 방법 0 (예측 성공 단어 / 총 단어)
def accuracy_0(text, label):
    print('0')
    text_words = text.split()
    ground_truth_words = label.split()

    correct_words = sum(1 for word1, word2 in zip(text_words, ground_truth_words) if word1 == word2)
    total_words = len(ground_truth_words)

    accuracy = correct_words / total_words
    return accuracy


# 텍스트 유사도 계산: 방법 1 (Jaccard Similarity)
def accuracy_1(text, label):
    print('1')
    intersection_cardinality = len(set.intersection(*[set(text), set(label)]))
    union_cardinality = len(set.union(*[set(text), set(label)]))
    accuracy = intersection_cardinality / float(union_cardinality)

    return accuracy


# 텍스트 유사도 계산: 방법 2 (Sequence Matcher)
def accuracy_2(text, label):
    print('2')
    answer_bytes = bytes(label, 'utf-8')
    input_bytes = bytes(text, 'utf-8')
    answer_bytes_list = list(answer_bytes)
    input_bytes_list = list(input_bytes)

    sm = difflib.SequenceMatcher(None, answer_bytes_list, input_bytes_list)
    similar = sm.ratio()

    return similar


# 텍스트 유사도 계산: 방법 3 (Levenshtein Distance)
def accuracy_3(text, label):
    print('3')
    similarity = float(fuzz.ratio(text, label)) / 100
    return similarity


# 텍스트 유사도 계산: 방법 4 (Cosine Similarity)
def accuracy_4(text, label):
    print('4')
    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b|[\w\W]+")
    tfidf_matrix = vectorizer.fit_transform([text, label])

    # 2개 텍스트 간의 cosine similarity 계산
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return similarity[0][0]


# 텍스트 유사도 계산: 방법 5 (Euclidean Distance)
def accuracy_5(text, label):
    print('5')
    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text, label])

    # 정규화
    tfidf_normalized = tfidf_matrix / np.sum(tfidf_matrix)

    # 2개 텍스트 간의 euclidean similarity 계산
    similarity = euclidean_distances(tfidf_normalized[0], tfidf_normalized[1])

    return 1 - similarity[0][0]


# 텍스트 유사도 계산: 방법 6 (Manhattan Distance)
def accuracy_6(text, label):
    print('6')
    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text, label])

    # 정규화
    tfidf_normalized = tfidf_matrix / np.sum(tfidf_matrix)

    # 2개 텍스트 간의 manhattan similarity 계산
    similarity = manhattan_distances(tfidf_normalized[0], tfidf_normalized[1])

    return 1 - similarity[0][0]


path = 'test_data/'
lang = 'vie_train'

img_list = [file for file in os.listdir(path) if re.search(r'\.(png|jpg|jpeg|tif)$', file, re.I)]
label_list = [file for file in os.listdir(path) if re.search(r'\.(txt|gt.txt)$', file, re.I)]

recog_text_list = []
label_text_list = []
acc_list = []

# for label in label_list:
#     with open(path + label, 'r', encoding="utf-8") as file:
#         text = file.read()
#         label_text_list.append(text)

# for img in img_list:
#     predict_text = pytesseract.image_to_string(Image.open(path + img), lang=lang, config='--psm 13')
#
#     img_name = img.rsplit('.', 1)[0]
#     if os.path.exists(path + img_name + '.txt'):
#         label = path + img_name + '.txt'
#     elif os.path.exists(path + img_name + '.gt.txt'):
#         label = path + img_name + '.gt.txt'
#     else:
#         print(f"Label file of {img} is not exist")
#         exit(1)
#
#     with open(label, 'r', encoding="utf-8") as file:
#         label_text = file.read()
#     print('Label:', label_text, ' - Predict:', predict_text)
#     acc_list.append(accuracy_6(predict_text, label_text))

# print(img_list)
# print(acc_list)
# print(statistics.mean(acc_list))

# test_txt = pytesseract.image_to_string(Image.open(path + '0.jpg'), lang='vie_train', config='--psm 13')
#
# with open(path + '0.txt', 'r', encoding="utf-8") as file:
#     test_label = file.read()
# print('Label:'+test_label)
# print('Predict:'+test_txt.rstrip('\n'))
print(accuracy_4('le quang trugn','le quang trung'))
