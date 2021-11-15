# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 15:24:46 2021

@author: xboxk
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


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

points = []
flag = False
def click(event, x, y, flags, param):
    global flag, tam
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.pop(-1)
    imprimir_2 = draw_image.copy()
    for pt in points:
        cv2.circle(imprimir_2, (pt[0]*tam,pt[1]*tam), 5, [255,0,0], -1)
        if(imprimir_2.shape[0]>1500):
            imprimir_image("img",imprimir_2,3)
        elif(imprimir_2.shape[0]>1000):
            imprimir_image("img",imprimir_2,2)
        else:
            imprimir_image("img",imprimir_2,1)

def guardar_datos():
    global keypoints,descriptors,points, tam,nombre_array,nombre_archivo
    coor_key = []
    salida = np.zeros((len(points),128))
    points = np.array(points)*tam
    
    
    for poi in keypoints:
        coor = poi.pt
        coor_key.append([coor[0],coor[1]])
    coor_key= np.array(coor_key)

    for i in range(points.shape[0]):
        pos= np.argmin(np.sum((points[i,:]-coor_key)**2,1))
        salida[i,:]= descriptors[pos]
    points = []
    try:
        os.mkdir("cuchilla/"+nombre_array)
    except : 
        xd = 1
    np.save("cuchilla/"+nombre_array+nombre_archivo+".npy", salida)
    
        
        
        

        

def funcion(Img_lectura, external_variable= 0):
    global draw_image, tam, keypoints, descriptors
    cv2.namedWindow("img")
    cv2.setMouseCallback("img", click)
    Img_lectura = cv2.cvtColor(Img_lectura, cv2.COLOR_BGR2GRAY)
    h,w = Img_lectura.shape
    
    hist = cv2.calcHist([Img_lectura], [0],None , [255], [0, 255])

    # plt.plot([0,255],[h*w*0.03,h*w*0.03])
    # plt.plot(hist)
    # plt.title("Histograma imagen")
    # plt.show()



    if(hist.max()> h*w*0.03):
        # Extrae la posicion del punto maximo del histograma
        max_pos = int(hist.argmax())
    
        # Extrayendo la mascara apartir del punto maximo encontrado
        lim_inf = (max_pos - 10)
        lim_sup = (max_pos + 10)
        mask = cv2.inRange(Img_lectura, lim_inf, lim_sup)
        mask = cv2.bitwise_not(mask)
        
        
        mask_flood = np.zeros((h + 2, w + 2), np.uint8)
        mascara_floodfill = mask.copy()
        cv2.floodFill(mascara_floodfill, mask_flood, (0, h//2), 255)
        mascara_floodfill = cv2.bitwise_not(mascara_floodfill)
        mascara_floodfill = cv2.bitwise_or(mascara_floodfill,mask)
        
        
        kernel = np.ones((11,11),np.uint8)
        mask = cv2.morphologyEx(mascara_floodfill, cv2.MORPH_OPEN, kernel)
        
        
        
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
    
    
    bordes = cv2.Laplacian(bordes,cv2.CV_64F)
    bordes = cv2.convertScaleAbs(bordes)
    _ , bordes = cv2.threshold(bordes ,1 , 255, cv2.THRESH_BINARY)
    
  

    
    
    
    

    lim = 150
    ret, labels = cv2.connectedComponents(bordes)
    new_ret = np.arange(0,ret).tolist()
    for i in range(ret):
        vec_true = (labels==i)
        tam_borde = np.where(vec_true)[0].shape[0]
        if(tam_borde<lim):
            labels[vec_true]= 0
            new_ret[i]= -1

    i = 0
    while(i<len(new_ret)):
        if(new_ret[i]==-1):
            new_ret.pop(i)
        else:
            i+=1

    ret = len(new_ret)
        
    

    borde_filtrado = np.zeros_like(bordes)
    borde_filtrado[labels==(new_ret[0])]=255
    
    borde_filtrado = cv2.bitwise_not(borde_filtrado)
    
    draw_image = cv2.merge((borde_filtrado,borde_filtrado,borde_filtrado))
    
    sift = cv2.SIFT_create(nfeatures=1000)   # shift invariant feature transform
    keypoints, descriptors = sift.detectAndCompute(borde_filtrado, None)


    
    for point in keypoints:
        coor = point.pt
        x = int(coor[0])
        y = int(coor[1])
        image = cv2.circle(draw_image, (x,y), 5, [0,0,255], -1)
    imprimir= img_contrast
    #imprimir_2 =cv2.bitwise_and(Img_lectura,mask_new) 
    imprimir_2 = draw_image
    imprimir_3 = borde_filtrado 
    #cv2.imwrite("Image_ocluida.png",draw_image)
    if(Img_lectura.shape[0]>1500):
        tam= 3
        imprimir_image("Nombre",imprimir,3)
        imprimir_image("img",imprimir_2,3)
        imprimir_image("img_2",imprimir_3,3)
    elif(Img_lectura.shape[0]>1000):
        tam= 2
        imprimir_image("Nombre",imprimir,2)
        imprimir_image("img",imprimir_2,2)
        imprimir_image("img_2",imprimir_3,2)
    else:
        tam = 1
        imprimir_image("Nombre",imprimir,1)
        imprimir_image("img",imprimir_2,1)
        imprimir_image("img_2",imprimir_3,1)


class pruebas():
     def func(self,modo, image = None,var = 0):
         if(modo == 1):
             Img_lectura = leer_imagen(1,1)
             funcion(Img_lectura)
         if(modo == 2):
             funcion(image,var)
            
     def visualizar(self,tiempo):
        global nombre_array,nombre_archivo
        cont_folder = 2
        flag = False
        for i in range(82):
            cont_image = 1
            cambio = True
            cont_var = 0
            print(cont_folder)
            while(True):
                
                try:
                    maleta = "B"+ string_zeros(3-decenas(cont_folder)) +str(cont_folder)
                    N_imagen = string_zeros(3-decenas(cont_image))+str(cont_image)
                    Img_read = cv2.imread("Data_set/Baggages/"+(maleta)+"/"+(maleta)+"_"+(N_imagen)+".png",cv2.IMREAD_COLOR)
                    nombre_array = (maleta)
                    nombre_archivo ="/"+(maleta)+"_"+(N_imagen)
                    if(cambio == True):
                        self.func(2,Img_read,cont_var)
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
                        if key == ord("b"):
                            cont_var+=1
                            cambio = True
                        if key == ord("v"):
                            cont_var-=1
                            cambio = True
                            if(cont_var<0):cont_var=0
                        if key == ord("x"):
                            guardar_datos()
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