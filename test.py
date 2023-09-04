# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.metrics.pairwise import cosine_similarity
# #
# # def accuracy_3(text, label):
# #     # Tạo một TfidfVectorizer và không loại bỏ các từ ngừng
# #     vectorizer = TfidfVectorizer(stop_words=None, token_pattern= r"(?u)\b\w+\b|[\w\W]+") #r"(?u)\b\w+\b|\s+|\.")
# #     tfidf_matrix = vectorizer.fit_transform([text, label])
# #
# #     # Tính cosine similarity giữa hai văn bản
# #     similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
# #
# #     return similarity[0][0]
# #
# # similarity = accuracy_3('#', '#')
# # print(f"Cosine Similarity: {similarity}")
#
import re

# Chuỗi đầu vào
text = "le quang trung"

# Sử dụng biểu thức chính quy \w
pattern_1 = r'(?u)\b\w\w+\b'

# Tìm tất cả các khớp với \w
matches_1 = re.findall(pattern_1, text)
print("Matches for \\w:", matches_1)

# Sử dụng biểu thức chính quy \w\w+
pattern_2 = r'\w\w+'

# Tìm tất cả các khớp với \w\w+
matches_2 = re.findall(pattern_2, text)
print("Matches for \\w\\w+:", matches_2)

pattern_3 = r"(?u)\b\w\w+\b"

# Tìm tất cả các khớp với \w\w+
matches_3 = re.findall(pattern_3, text)
print("Matches for \\w\\w+:", matches_3)


