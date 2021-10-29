# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 18:09:08 2021

@author: xboxk
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt



def imprimir_image(nombre,imagen,factor):
    if(len(imagen.shape)>2):
        imageOut=imagen[::factor, ::factor,:]
    else:
        imageOut=imagen[::factor, ::factor]
    cv2.imshow(nombre,imageOut)


def decenas(num):
    decenas = num//10
    if decenas >= 2:decenas = 1
    return decenas

def string_zeros(num):
    zeros = ""
    for i in range(num):
        zeros+= str(0) 
    return zeros


        
def leer_imagen(N_folder,N_imagen):
     try:
        maleta = "B"+ string_zeros(3-decenas(N_folder)) +str(N_folder)
        N_imagen = string_zeros(3-decenas(N_imagen))+str(N_imagen)
        Img_read = cv2.imread("D:/Documentos/8tvo_semestre/Prosesamiento_de_imagenes_vision/Proyecto/Data_set/Baggages/"+(maleta)+"/"+(maleta)+"_"+(N_imagen)+".png",cv2.IMREAD_COLOR)
        return Img_read
     except:
         print("No se encontro imagen")
         return -1

def find_peaks(funcion, nhood, accumulator_threshold, N_peaks):
        done = False
        acc_copy = funcion.copy()
        nhood_center = (nhood - 1) / 2
        peaks = []
        while not done:
            pos = acc_copy.argmax()
            if acc_copy[pos] >= accumulator_threshold:
                peaks.append(pos)

                lim_1 = np.uint32(np.max([pos - nhood_center, 0]))
                lim_2 = np.uint32(np.min([pos + nhood_center, acc_copy.shape[0] - 1]) + 1)
                
                acc_copy[lim_1:lim_2]=0
                done = np.array(peaks).shape[0] == N_peaks
            else:
                done = True

        return peaks
  
def paint(img_gray, nhood, color,img_draw,color_pintar):
    # imagen tiene que se blanco y negro
    
    
    lim_inf = np.array(np.uint8(np.max([color - nhood//2, 0])))
    lim_sup = np.array(np.uint8(np.min([color + nhood//2, 254]) + 1))

    mask = cv2.inRange(img_gray, lim_inf, lim_sup)
    
    n_mask = cv2.bitwise_not(mask)
    
    img_draw= cv2.bitwise_and(img_draw,cv2.merge((n_mask,n_mask,n_mask)))
    
    
    
    img_color = np.ones_like(img_draw)
    img_color[:,:,:] = color_pintar
    
    
    img_color= cv2.bitwise_and(img_color,cv2.merge((mask,mask,mask)))
    

    
    img_draw = cv2.bitwise_or(img_color,img_draw)


    
    return img_draw
    
    
def dectetion_descriptor(image):
    image_draw = cv2.merge((image,image,image))
    sift = cv2.SIFT_create(nfeatures=1000)   # shift invariant feature transform
    keypoints_1, descriptors_1 = sift.detectAndCompute(image, None)
    points_1= []
    [points_1.append(np.int32(idx.pt)) for idx in keypoints_1]
    for i in range(len(points_1)): 
        
        cv2.circle(image_draw, points_1[i], 5, [0,0,255], -1)
    return image_draw
         

#visualizar()

def funcion(Img_lectura):
    Img_lectura = cv2.cvtColor(Img_lectura, cv2.COLOR_BGR2GRAY)
    h,w = Img_lectura.shape
    
    hist = cv2.calcHist([Img_lectura], [0],None , [255], [0, 255])

    if(hist.max()> h*w*0.03):
        # Extrae la posicion del punto maximo del histograma
        max_pos = int(hist.argmax())
    
        # Extrayendo la mascara apartir del punto maximo encontrado
        lim_inf = (max_pos - 10)
        lim_sup = (max_pos + 10)
        mask = cv2.inRange(Img_lectura, lim_inf, lim_sup)
        mask = cv2.bitwise_not(mask)
        
        kernel = np.ones((11,11),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        img_mask = cv2.bitwise_and(Img_lectura,mask)
        hist = cv2.calcHist([img_mask], [0],mask , [255], [0, 255])
    else:
        mask = np.ones_like(Img_lectura)
        img_mask = Img_lectura
        hist = cv2.calcHist([img_mask], [0],None , [255], [0, 255])
    
    img_contrast = np.power(img_mask,1.1).clip(0,255).astype(np.uint8)
    
    
    neig = 15
    
    imagen_gausiana_1 = cv2.GaussianBlur(img_contrast,(neig,neig),sigmaX= 3,sigmaY=3)
    imagen_gausiana_2 = cv2.GaussianBlur(img_contrast,(neig,neig),sigmaX= 4,sigmaY=4)
    bordes = imagen_gausiana_2-imagen_gausiana_1
        
    kernel= np.ones((3,3),np.uint8)
    bordes=cv2.erode(bordes,kernel)
    bordes=cv2.dilate(bordes,kernel)
    
    _ , bordes = cv2.threshold(bordes ,200 , 255, cv2.THRESH_BINARY)
    
    #bordes = cv2.Canny(image=bordes, threshold1=100, threshold2=255)
    
    #kernel= np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    #bordes = cv2.filter2D(bordes,-1,kernel)
    
    bordes = cv2.Laplacian(bordes,cv2.CV_64F)
    bordes = cv2.convertScaleAbs(bordes)
    _ , bordes = cv2.threshold(bordes ,1 , 255, cv2.THRESH_BINARY)
    
    #_ , bordes = cv2.threshold(bordes ,1 , 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bordes, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    draw_image= cv2.merge((Img_lectura,Img_lectura,Img_lectura))
    
    for idx, i in enumerate(contours):
        if(len(i)>100):
            cv2.drawContours(draw_image, i, -1, (0,0,255), 3)
    
    
    
    
    
    
    #kernel= np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    #edge = cv2.filter2D(bordes,-1,kernel)
    
    
    #edges = cv2.Canny(image=img_contrast, threshold1=100, threshold2=250)
    #kernel= np.array([[0,0,0],[1,1,1],[0,0,0]],np.uint8)
    
    #edges=cv2.erode(edges,kernel)
    #mascara_final=cv2.dilate(mascara_final,kernel_linea)
    
    #_ , edges = cv2.threshold(edges ,10 , 255, cv2.THRESH_BINARY)

    
    

    """
    alpha = 0.0005  
    nhood= 20
    threshold= h*w*alpha
    N_peaks = 20
    peaks = find_peaks(hist, nhood, threshold, N_peaks)
    

    beta = float(hist[peaks[0]])*0.5

    plt.plot([0,255],[h*w*alpha,h*w*alpha])
    plt.plot([0,255],[beta,beta])
    plt.scatter(peaks,hist[peaks],c =np.ones(len(peaks)) )
    plt.plot(hist)
    plt.show()

    img_draw= np.ones((img_mask.shape[0],img_mask.shape[1],3),np.uint8)*255
    
    mask_aux = np.zeros((img_mask.shape[0],img_mask.shape[1]),np.uint8)*255
    for color_pos in (peaks):
        if(hist[color_pos]<beta):
            lim_inf = np.array(np.uint8(np.max([color_pos - nhood//2, 0])))
            lim_sup = np.array(np.uint8(np.min([color_pos + nhood//2, 254]) + 1))
        
            mask_aux = cv2.bitwise_or(cv2.inRange(img_mask, lim_inf, lim_sup),mask_aux)
            
            img_draw = paint(img_mask, nhood, color_pos,img_draw, np.uint8((np.random.rand(1,3)*255)))
    
    mask_new = cv2.bitwise_and(mask_aux,mask)
    """
    
    #img_contrast = np.power(img_mask,1.1).clip(0,255).astype(np.uint8)
    #edges = cv2.Canny(image=img_contrast, threshold1=1, threshold2=250)
    """ 
    imagen_blur=cv2.blur(img_mask,(11,11),0)

    img_contrast = np.power(imagen_blur,1.1).clip(0,255).astype(np.uint8)
    
    kernel= np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    edges = cv2.filter2D(img_mask,-1,kernel)
    _ , edges = cv2.threshold(edges ,20 , 255, cv2.THRESH_BINARY)
    """
    
    imprimir= img_contrast
    #imprimir_2 =cv2.bitwise_and(Img_lectura,mask_new) 
    imprimir_2 = bordes
    imprimir_3 = draw_image
    if(Img_lectura.shape[0]>1500):
        imprimir_image("Nombre",imprimir,3)
        imprimir_image("img",imprimir_2,3)
        imprimir_image("img_2",imprimir_3,3)
    elif(Img_lectura.shape[0]>1000):
        imprimir_image("Nombre",imprimir,2)
        imprimir_image("img",imprimir_2,2)
        imprimir_image("img_2",imprimir_3,2)
    else:
        imprimir_image("Nombre",imprimir,1)
        imprimir_image("img",imprimir_2,1)
        imprimir_image("img_2",imprimir_3,1)


class pruebas():
     def func(self,modo, image = None):
         if(modo == 1):
             Img_lectura = leer_imagen(1,1)
             funcion(Img_lectura)
         if(modo == 2):
             funcion(image)
            
     def visualizar(self,tiempo):
        cont_folder = 1
        flag = False
        for i in range(82):
            cont_image = 1
            cambio = True
            print(cont_folder)
            while(True):
                
                try:
                    maleta = "B"+ string_zeros(3-decenas(cont_folder)) +str(cont_folder)
                    N_imagen = string_zeros(3-decenas(cont_image))+str(cont_image)
                    Img_read = cv2.imread("Data_set/Baggages/"+(maleta)+"/"+(maleta)+"_"+(N_imagen)+".png",cv2.IMREAD_COLOR)
                    
                    if(cambio == True):
                        self.func(2,Img_read)
                        cambio = False
                    
                    #imprimir_image("Imagen", Img_read,3)
                    if(tiempo != -1):
                        if cv2.waitKey(tiempo) & 0xFF == 27:
                            cv2.destroyAllWindows()
                            flag = True
                            
                            break
                        cont_image+=1
                        cambio = True
                    else:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("x"):
                            cont_image+=1
                            cambio = True
                        if key == ord("c"):
                            cv2.destroyAllWindows()
                            break
                        if key == ord("z"):
                            flag = True;
                            cv2.destroyAllWindows()
                            break
                           
                    
                except Exception as e:
                    print(e)
                    break
            if(flag): break
            cont_folder +=1
             
pru = pruebas()
pru.visualizar(-1)
#pru.func(1)




cv2.waitKey(0)
