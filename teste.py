import pytesseract
import cv2 
from matplotlib import pyplot as plt


img = cv2.imread('teste1.jpeg')
plt.imshow(img)


config_tesseract = '--tessdata-dir tessdata --psm 7'
texto = pytesseract.image_to_string(img, lang='por', config=config_tesseract)
print(texto)