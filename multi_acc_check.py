# 필요한 패키지 가져 오기
import os
import pytesseract
import argparse
import glob
import statistics
import numpy as np
from PIL import Image
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from fuzzywuzzy import fuzz

ap = argparse.ArgumentParser(description='Evaluate Tesseract accuracy')

ap.add_argument('-i', '--image', nargs='+', type=str, required=True,
                help="path to image")
ap.add_argument('-l', '--language', type=str, default='eng',
                help="recognition language")
ap.add_argument('-psm', '--pageseg', type=int, default=3,
                help="image segmentation mode")
ap.add_argument('-a', '--accuracy', type=int, default=1,
                help="accuracy checking method")
ap.add_argument('-v', '--verbose', action='store_true',
                help="show details for every single image")

# Convert Namespace(result of ap.parse_args() to dict type)
args = vars(ap.parse_args())


# 텍스트 유사도 계산: 방법 0 (예측 성공 단어 / 총 단어)
def accuracy_0(text, label):
    text_words = text.split()
    ground_truth_words = label.split()

    correct_words = sum(1 for word1, word2 in zip(text_words, ground_truth_words) if word1 == word2)
    total_words = len(ground_truth_words)

    accuracy = correct_words / total_words
    return accuracy


# 텍스트 유사도 계산: 방법 1 (Jaccard Similarity)
def accuracy_1(text, label):
    intersection_cardinality = len(set.intersection(*[set(text), set(label)]))
    union_cardinality = len(set.union(*[set(text), set(label)]))
    accuracy = intersection_cardinality / float(union_cardinality)

    return accuracy


# 텍스트 유사도 계산: 방법 2 (Sequence Matcher)
def accuracy_2(text, label):
    answer_bytes = bytes(label, 'utf-8')
    input_bytes = bytes(text, 'utf-8')
    answer_bytes_list = list(answer_bytes)
    input_bytes_list = list(input_bytes)

    sm = difflib.SequenceMatcher(None, answer_bytes_list, input_bytes_list)
    similar = sm.ratio()

    return similar


# 텍스트 유사도 계산: 방법 3 (Levenshtein Distance)
def accuracy_3(text, label):
    similarity = float(fuzz.ratio(text, label)) / 100
    return similarity


# 텍스트 유사도 계산: 방법 4 (Cosine Similarity)
def accuracy_4(text, label):
    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer(token_pattern=r'\S+')  # r'\S+|[\W]+')    #r"(?u)\b\w\w+\b|[\w\W]+")
    tfidf_matrix = vectorizer.fit_transform([text, label])

    # 2개 텍스트 간의 cosine similarity 계산
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return similarity[0][0]


# 텍스트 유사도 계산: 방법 5 (Euclidean Distance)
def accuracy_5(text, label):
    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer(token_pattern=r'\S+')
    tfidf_matrix = vectorizer.fit_transform([text, label])

    # 정규화
    tfidf_normalized = tfidf_matrix / np.sum(tfidf_matrix)

    # 2개 텍스트 간의 euclidean similarity 계산
    similarity = euclidean_distances(tfidf_normalized[0], tfidf_normalized[1])

    return 1 - similarity[0][0]


# 텍스트 유사도 계산: 방법 6 (Manhattan Distance)
def accuracy_6(text, label):
    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer(token_pattern=r'\S+')
    tfidf_matrix = vectorizer.fit_transform([text, label])

    # 정규화
    tfidf_normalized = tfidf_matrix / np.sum(tfidf_matrix)

    # 2개 텍스트 간의 manhattan similarity 계산
    similarity = manhattan_distances(tfidf_normalized[0], tfidf_normalized[1])

    return 1 - similarity[0][0]


acc_list = []

for image in args['image']:
    files = glob.glob(image)

    if not files:
        print(f"{image} is not exist")
        exit(1)

    for idx, img in enumerate(files):
        predict_text = pytesseract.image_to_string(Image.open(img), lang=args['language'],
                                                   config=f"--psm {args['pageseg']}").rstrip('\n')

        img_name = img.rsplit('.', 1)[0]
        if os.path.exists(img_name + '.txt'):
            label = img_name + '.txt'
        elif os.path.exists(img_name + '.gt.txt'):
            label = img_name + '.gt.txt'
        else:
            print(f"Label file of {img} is not exist")
            exit(1)

        with open(label, 'r', encoding="utf-8") as file:
            label_text = file.read()

        if args['verbose']:
            print('\n   ' + img + ':')
            print(f"{'-Label:':<10}", label_text)
            print(f"{'-Predict:':<10}", predict_text)

        if args['accuracy'] == 0:
            acc = accuracy_0(predict_text, label_text)
        elif args['accuracy'] == 1:
            acc = accuracy_1(predict_text, label_text)
        elif args['accuracy'] == 2:
            acc = accuracy_2(predict_text, label_text)
        elif args['accuracy'] == 3:
            acc = accuracy_3(predict_text, label_text)
        elif args['accuracy'] == 4:
            acc = accuracy_4(predict_text, label_text)
        elif args['accuracy'] == 5:
            acc = accuracy_5(predict_text, label_text)
        elif args['accuracy'] == 6:
            acc = accuracy_6(predict_text, label_text)
        else:
            acc = accuracy_1(predict_text, label_text)

        if args['verbose']:
            print(f"{'-Accuracy:':<10}", acc)

        acc_list.append(acc)

# if args['verbose']:
#     print(f"\n{'-Acc List:':<10}", acc_list)
print('--------DONE--------')
print(f"{'-Avg Acc:':<10}", statistics.mean(acc_list))
