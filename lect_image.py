
import cv2
import numpy as np
import sys
if(sys.platform !="win32"):
    import matplotlib
    matplotlib.use('TkAgg')
from matplotlib import pyplot as pl

def imprimir_image(nombre,imagen,factor):
    imageOut = cv2.resize(imagen,(imagen.shape[0]//factor,imagen.shape[1]//factor), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(nombre,imageOut)
    
    




path_file= "/media/andres/Datos/Documentos/8tvo_semestre/Prosesamiento_de_imagenes_vision/Proyecto/Data_set/Baggages/B0014/B0014_0002.png"
image= cv2.imread(path_file)

img_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img_gray = 255-img_gray



## ley de potencias
img_contrast = np.power(img_gray,1.3).clip(0,255).astype(np.uint8)
##Ecualizacion de histograma
img_ecualize = cv2.equalizeHist(img_gray)
## Ecualizacion adaptativa 
clahe = cv2.createCLAHE(cliplimit=2.0,tileGridsize=(8,8))

hist =cv2.calcHist([img_gray], [0],None , [255], [0, 255])
hist_2 =cv2.calcHist([img_contrast], [0],None , [255], [0, 255])
hist_3 =cv2.calcHist([img_ecualize], [0],None , [255], [0, 255])

pl.plot(hist)
pl.show()
pl.plot(hist_2)
pl.show()
pl.plot(hist_3)
pl.show()




imprimir_image("imagen",img_gray,6)
imprimir_image("imagen ley potencia",img_contrast,6)
imprimir_image("imagen ecualizada",img_ecualize,6)
cv2.waitKey(0)



