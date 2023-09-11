import cv2
import numpy as np

# Đường dẫn đến hình ảnh
image_path = 'multiline_data/22.jpg'

# Đọc hình ảnh
image = cv2.imread(image_path)

# Chuyển đổi hình ảnh sang ảnh đen-trắng
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Nhị phân hóa ảnh để làm nổi bật văn bản trên nền
_, binary = cv2.threshold(gray, 20, 100, cv2.THRESH_BINARY)

# Tìm các đường viền trong ảnh
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Tạo một mặt nạ trắng có cùng kích thước với ảnh
mask = np.ones_like(image) * 255

# Vẽ chữ lên mặt nạ
cv2.drawContours(mask, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

# Sử dụng mặt nạ để tách văn bản ra khỏi nền
result = cv2.bitwise_and(image, mask)

# Hiển thị hình ảnh kết quả
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
