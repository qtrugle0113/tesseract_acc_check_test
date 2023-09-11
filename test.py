import cv2
import os



path = 'multiline_data/'

img_color = cv2.imread(path, cv2.IMREAD_COLOR)

if img_color is None:
    print("Read image fail")
    exit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 11)


cv2.imshow('Result', img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite('test_hw_out/' + path, img_thresh)

cv2.imwrite(path.replace('.jpg', '_cvt.jpg'), img_thresh)
