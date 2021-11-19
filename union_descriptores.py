# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 07:49:54 2021

@author: xboxk
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def decenas(num):
    decenas = num//10
    if decenas >= 2:decenas = 1
    return decenas

def string_zeros(num):
    zeros = ""
    for i in range(num):
        zeros+= str(0) 
    return zeros


def borrar_vacios(datos):
    datos = datos[:]
    cont = 0
    while(cont <len(datos)):
        if(datos[cont].shape[0]==0):
            datos.pop(cont)
        else:
            cont+=1
    return datos

def visualizar():
    puntos = []
    cont_folder =2
    flag = False
    for i in range(2,82):
        cont_image = 1
        cambio = True
        cont_var = 0 
        while(True):
            
            try:
                maleta = "B"+ string_zeros(3-decenas(cont_folder)) +str(cont_folder)
                N_imagen = string_zeros(3-decenas(cont_image))+str(cont_image)
                puntos.append(np.load("pistola/"+(maleta)+"/"+(maleta)+"_"+(N_imagen)+".npy"))

                
                cont_image+=1
                       
                
            except Exception as e:
                break
        if(flag): break
        cont_folder +=1
    return puntos
        
        
        
datos= visualizar()
datos_bien=(borrar_vacios(datos))

longitud = 0
for i in (datos_bien):
    longitud+= i.shape[0]

vec_out = np.zeros((longitud,128))

cont = 0
for i,array in enumerate(datos_bien):
    
    vec_out[cont:cont+array.shape[0],:]=array
    cont += array.shape[0]

# Escalar datos
scaler = StandardScaler()
scaler.fit(vec_out)
X = scaler.transform(vec_out)

silueta = []

#for i in range(153,1000):
if(1):
    i = 35
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    kmeans.fit(X)
    
    centroide = kmeans.cluster_centers_
    
    
    centroide = scaler.inverse_transform(centroide)
    
    labels_pred= kmeans.fit_predict(X)
    
    silueta.append(silhouette_score(X, labels_pred, metric='euclidean'))

np.save("pistola.npy",vec_out)