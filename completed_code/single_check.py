# 필요한 패키지 가져 오기
import os
import pytesseract
import argparse
import numpy as np
from PIL import Image
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
from fuzzywuzzy import fuzz

ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', type=str, required=True,
                help="path to image")
ap.add_argument('-l', '--language', type=str, default='eng',
                help="recognition language")
ap.add_argument('-psm', '--pageseg', type=int, default=3,
                help="image segmentation mode")
ap.add_argument('-a', '--accuracy', type=int, default=1,
                help="accuracy checking method")

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


text = pytesseract.image_to_string(Image.open(args['image']), lang=args['language'],
                                   config=f"--psm {args['pageseg']}").rstrip('\n')  # 맨 끝에 있는 '\n' 제거

img_name = args['image'].rsplit('.', 1)[0]
if os.path.exists(img_name + '.txt'):
    label_path = img_name + '.txt'
elif os.path.exists(img_name + '.gt.txt'):
    label_path = img_name + '.gt.txt'
else:
    print(f"Label file of {args['image']} is not exist")
    exit(1)

with open(label_path, 'r', encoding='utf-8') as file:
    label = file.read()

if args['accuracy'] == 0:
    acc = accuracy_0(text, label)
elif args['accuracy'] == 1:
    acc = accuracy_1(text, label)
elif args['accuracy'] == 2:
    acc = accuracy_2(text, label)
elif args['accuracy'] == 3:
    acc = accuracy_3(text, label)
elif args['accuracy'] == 4:
    acc = accuracy_4(text, label)
elif args['accuracy'] == 5:
    acc = accuracy_5(text, label)
elif args['accuracy'] == 6:
    acc = accuracy_6(text, label)
else:
    acc = accuracy_1(text, label)

print(f"{'-Label:':<10}", label)
print(f"{'-Predict:':<10}", text)
print(f"{'-Accuracy:':<10}", acc)
