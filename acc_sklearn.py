from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def text_similarity(file1_path, file2_path):
    # .txt 파일 내용 읽기
    with open(file1_path, 'r', encoding='utf-8') as file1:
        text1 = file1.read()
    with open(file2_path, 'r', encoding='utf-8') as file2:
        text2 = file2.read()

    # 텍스트에 대한 TF-IDF 벡터 만들기
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # 2개 텍스트 간의 cosine similarity 계산
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

    return similarity[0][0]

