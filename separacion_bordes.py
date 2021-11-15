# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:46:47 2021

@author: xboxk
"""
import cv2
import numpy as np


def imprimir_image(nombre,imagen,factor):
    if(len(imagen.shape)>2):
        imageOut=imagen[::factor, ::factor,:]
    else:
        imageOut=imagen[::factor, ::factor]
    cv2.imshow(nombre,imageOut)


imagen  = cv2.imread("D:/Documentos/8tvo_semestre/Prosesamiento_de_imagenes_vision/Proyecto/Image_ocluida.png")

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

kernel = np.ones((9,9),np.uint8)
dilation = cv2.dilate(imagen,kernel,iterations = 1)
erode = cv2.erode(dilation,kernel,iterations = 1)



h,w = imagen.shape[:2]


    

h,w = imagen.shape[:2]

flood = np.zeros((h + 2, w + 2), np.uint8)
floodfill = erode.copy()
cv2.floodFill(floodfill, flood, (0, 0), 255)


puntos = np.where(imagen)


imprimir_image("img",imagen,2)
imprimir_image("img2",floodfill,2)
cv2.waitKey(0)
