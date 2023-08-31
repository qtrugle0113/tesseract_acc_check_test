import os
import random
import shutil

# Đường dẫn đến thư mục A và B
folderA_path = 'hand_write'
folderB_path = 'hw_data'

# Lấy danh sách tệp trong thư mục A
file_list = [file for file in os.listdir(folderA_path) if file.lower().endswith('.png')]

# Chọn ngẫu nhiên 10 tệp từ danh sách
random_files = random.sample(file_list, 1000)

# Duyệt qua danh sách tệp và sao chép sang thư mục B
for file_name in random_files:
    source_file = os.path.join(folderA_path, file_name)
    destination_file = os.path.join(folderB_path, file_name)
    shutil.copy(source_file, destination_file)

    text_file = os.path.join(folderA_path, file_name.replace('.png', '.txt'))
    destination_text_file = os.path.join(folderB_path, file_name.replace('.png', '.txt'))
    shutil.copy(text_file, destination_text_file)

print("Done")
