import time
import sys

total_img = 5

for idx in range(total_img):
    sys.stdout.write("\r" + f"{idx + 1}/{total_img}")
    sys.stdout.flush()
    time.sleep(1)  # Đợi 1 giây trước khi qua lượt lặp tiếp theo

# In ra dòng mới để không bị ghi đè lên
print()