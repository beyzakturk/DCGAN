# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:45:30 2022

@author: Beyza
"""

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Reshape
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import UpSampling2D, Conv2D
from tensorflow.python.keras.layers import ELU
from tensorflow.python.keras.layers import Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.datasets import mnist

import os
from PIL import Image
#Python resim kütüphanesi

import numpy as np
import math

def combine_images(generated_images):
    total,width,height = generated_images.shape[:-1]
    cols = int(math.sqrt(total))
    rows = math.ceil(float(total)/cols)
    combined_image = np.zeros((height*rows, width*cols),
                              dtype=generated_images.dtype)

    for index, image in enumerate(generated_images):
        i = int(index/cols)
        j = index % cols
        combined_image[width*i:width*(i+1), height*j:height*(j+1)] = image[:, :, 0]
    return combined_image


def show_progress(epoch, batch, g_loss, d_loss, g_acc, d_acc):
    msg = "epoch: {}, batch: {}, g_loss: {}, d_loss: {}, g_accuracy: {}, d_accuracy: {}"
    print(msg.format(epoch, batch, g_loss, d_loss, g_acc, d_acc))

#Üretici yani Generator için fonksiyon tanımlayalım
#İçerisine öncelikler gürültü değeri, ilk nöründaki layer değeri son olarak da aktivasyon fonk. belirtilmezse kullanılcak olan.
def genarator(input_dim=100, units= 1024, activation='relu'):
    #ardışık model oluşturalım.
    model = Sequential()
    #şimdi bu modele layerlar ekleyelim. fully connected layers
    model.add(Dense(input_dim=input_dim, units=units))
    #Normalleştirme yöntemini belirtelim.
    model.add(BatchNormalization())
    model.add(Activation(activation))
    #dense layer'ı convolutional layer bağlayacağımız için 128*7*7 alır
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    #şimdi convolutional layera bağlayalım
    #Yeniden şekillendirme yapalım
    model.add(Reshape((7,7,128), input_shape =(128*7*7,)))
    #Boyutlar mnist uygun olması için 28*28*1 olmalı. Büyütelim
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64, (5,5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2,2)))
    model.add(Conv2D(64, (5,5), padding='same'))
    #Derinlik kaldı onu da filtre sayısı ile 1 yapalım
    model.add(Activation('tanh'))
    #tanh kullanılmasının sebebi daha sonrasında 255 çevirerek resim elde etmek
    print(model.summary())
    return model

#Discriminator içinde fonk oluşturalım
#içine filtre sayısı belirtiyoruz.
def discriminator(input_shape=(28,28,1), nb_filter=64):
    model= Sequential()
    #çıkan sonuç 1 e yakınsa gerçek.Öncelikle convolutional layer ekleyelim
    model.add(Conv2D(nb_filter, (5,5), strides=(2,2), padding='same', input_shape=input_shape))
    #strides kaydırılacak adım
    model.add(BatchNormalization())
    model.add(ELU())
    #Önceden denen sonuçlara göre en iyi sonuç elu
    model.add(Conv2D(2*nb_filter, (5,5), strides=(2,2)))
    model.add(BatchNormalization())
    model.add(ELU())
    #boyut 5*5*128 oldu.
    #dense layer ile bağlayalım. Önce boyut uyuşmazlığını ortadan kaldıralım
    model.add(Flatten()) #düzleştirme yaptık
    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(ELU())
    #overfitting için önlem alalım
    model.add(Dropout(0.5))
    #output değerini alalım
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model


batch_size = 32
num_epoch=50
learning_rate = 0.0002
image_path = 'images/'
if not os.path.exists(image_path):
    os.mkdir(image_path)
    

def train():
    (x_train, y_train), (_, _) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    #4 boyutlu olmalı
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    
    g= genarator()
    d= discriminator()
    
    optimize = Adam(lr= learning_rate, beta_1=0.5)
    #daha hızlı sonuç için momentum değerine 0.5 verdik
    d.trainable = True
    d.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer= optimize)
    d.trainable = False
    dcgan = Sequential([g,d])
    dcgan.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer= optimize)
    #d eğittikten sonra g den aldığımız resimleri tekrar d den geçiriyoruz ama d eğitmiyoruz. Çıkan sonuçlara göre sadece g eğitiyoruz.
    num_batches = x_train.shape[0] // batch_size
    gen_img = np.array([np.random.uniform(-1, 1, 100) for _ in range(49)])
    #gerçek mi sahte mi anlamak için etiketleri oluşturalım
    y_d_true = [1] * batch_size
    y_d_gen = [0] * batch_size
    y_g = [1] * batch_size
    
    for epoch in range(num_epoch):
        #parça parça eğitim yapalım
        for i in range(num_batches):
            x_d_batch = x_train[i*batch_size:(i+1)*batch_size]
            #0 dan 32ye kadar olanlar alıncak
            #genaratorun resim üretmesini sağlayalım
            #ilk parametre orta noktadır. Bu sayılar gen verdiğimiz gürültülerdir.
            x_g = np.array([np.random.normal(0, 0.5, 100) for _ in range(batch_size)])
            x_d_gen = g.predict(x_g)
            #şimdi d eğitimi yapalım. Önce gerçek resimleri veririz ve etiketleri belirtiriz.
            d_loss = d.train_on_batch(x_d_batch, y_d_true)
            #şimdi sahte resimler ile eğitim.
            d_loss = d.train_on_batch(x_d_gen, y_d_gen)
            #şimdi g üzerinde eğitim yapalım. Rastegele olan gürültüyü verdik. Etiket 1 olmasının nedeni gerçeğe yakın olsun.
            g_loss = dcgan.train_on_batch(x_g, y_g)
            #yazdırma işlemi yapalım.
            show_progress(epoch, i, g_loss[0], d_loss[0], g_loss[1], d_loss[1])
        #Eğitimin ne aşamada nasıl ürettiğini görelim.
        #birleştirilcek resimler için combine image
        image = combine_images(g.predict(gen_img))
        image = image * 127.5 + 127.5
        #şimdi resmi kayıt edelim. Renk değerlerini belirt. 
        Image.fromarray(image.astype(np.uint8)).save(image_path + "%03d.png" % (epoch))

#dosya çalıştığında çalışacak kısım. Yani dışardan çağırılınca çalışmayacak.
if __name__ == '__main__':
    train() 
    
    