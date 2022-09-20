import cv2
import numpy as np

class pre:
    def invert(self, img):
        return 255 - img
    
    def cinza(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def threshold(self, img, nv1=127, nv2=255):
        val, thresh = cv2.threshold(img, nv1, nv2, cv2.THRESH_BINARY)
        return thresh


    def threshold(self, img, nv1=127, nv2=255):
        val, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return otsu

    def limira_adaptativa(self, img):
        gray = self.cinza(img)
        val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return otsu
    
    def limira_adaptativa_gausiana(self, img):
        gray = self.cinza(img)
        val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return otsu

    def limira_adaptativa_media(self, img):
        gray = self.cinza(img)
        adapt_media = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 9)
        return adapt_media

    def resize(self, img, taxa=1.5):
        '''
        - INTER_NEAREST - uma interpolação de vizinho mais próximo. É muito usado por ser o mais rápido.
        - INTER_LINEAR - uma interpolação bilinear (é usada por padrão), boa no geral para aumentar e também pra diminuir imagens.
        - INTER_AREA - usa a relação de área de pixel. Pode ser um método preferido para a redução de imagens pois fornece resultados sem moiré (efeito geralmente indesejado na imagem). Mas quando a imagem é ampliada, é semelhante ao método INTER_NEAREST.
        - INTER_CUBIC - bicúbica (4x4 pixel vizinhos). Possui resultados melhores.
        - INTER_LANCZOS4 - interpolação Lanczos (8x8 pixel vizinhos). Dentre esses algoritmos, é o que apresenta resultados com a melhor qualidade.
        '''
        return cv2.resize(img, None, fx=taxa, fy=taxa, interpolation=cv2.INTER_CUBIC)

    def erosao(self, img, m=5):
        '''Acaba removendo alguns pixes nas bordas dos objetos'''
        return cv2.erode(img, np.ones((m, m), np.uint8))

    def erosao(self, img, m=5):
        '''Aumenta alguns pixes nas bordas dos objetos'''
        return cv2.dilate(self.erosao(img), np.ones((m, m), np.uint8))

    def erosao(self, img, m=5):
        return cv2.dilate(self.erosao(img), np.ones((m, m), np.uint8))
