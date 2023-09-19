import pytesseract
import os
from PIL import Image

path = 'snip_img/'

file_names = os.listdir(path)

for img in file_names:
    text = pytesseract.image_to_string(Image.open(path + img), lang='vie_train4', config='--psm 13').rstrip('\n')

    text_name = path + img.replace('.JPG', '.gt.txt')
    with open(text_name, 'w', encoding="utf-8") as file:
        file.write(text)

