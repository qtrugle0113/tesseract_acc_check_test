from sklearn.feature_extraction.text import TfidfVectorizer
sentences = ("이 요리 의 레시피 를 알려줘.",
        "이 요리 어떻게 만드는 지 알려줘.")
tfidf_vectorizer = TfidfVectorizer()
 # 문장 벡터화 하기(사전 만들기)
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

### 코사인 유사도 ###
from sklearn.metrics.pairwise import cosine_similarity
# 첫 번째와 두 번째 문장 비교
cos_similar = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print("코사인 유사도 측정")
print(cos_similar)

### 유클리디언 유사도 (두 점 사이의 거리 구하기) ###
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

## 정규화 ##
tfidf_normalized = tfidf_matrix/np.sum(tfidf_matrix)

##유클리디언 유사도##
euc_d_norm = euclidean_distances(tfidf_normalized[0:1],tfidf_normalized[1:2])
print("유클리디언 유사도 측정")
print(euc_d_norm)

### 맨하탄 유사도(격자로 된 거리에서의 최단거리) ###
from sklearn.metrics.pairwise import manhattan_distances
manhattan_d = manhattan_distances(tfidf_normalized[0:1],tfidf_normalized[1:2])
print("맨하탄 유사도 측정")
print(manhattan_d)