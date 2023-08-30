import os
import tensorflow as tf
import matplotlib as plt
import pandas as pd
import PIL
from tensorboard.plugins import projector

# File name change
# file_list = os.listdir('/home/ephemera/Project/Meow/Data/cat/')

# for i,j in enumerate(file_list):
#     os.rename('/home/ephemera/Project/Meow/Data/cat/'+str(j), '/home/ephemera/Project/Meow/Data/cat/'+f'cat_{i}.jpg')


# Image Pre-Processing
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/home/ephemera/Project/Meow/Data/',
    validation_split=0.3,
    subset='training',
    image_size=(100,100),
    batch_size=8,
    color_mode='grayscale',
    seed=642,
    label_mode='categorical'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory='/home/ephemera/Project/Meow/Data/',
    validation_split=0.3,
    subset='validation',
    image_size=(100,100),
    batch_size=8,
    color_mode='grayscale',
    seed=642,
    label_mode='categorical'
)

#ResNet-18
# Input Layer
input1 = tf.keras.Input(shape=(100,100,1))

# Convolution Block A
conv_block1 = tf.keras.layers.Conv2D(64, (7,7), strides=2, padding='same')(input1)
conv_block2 = tf.keras.layers.BatchNormalization()(conv_block1)
conv_block3 = tf.keras.layers.ReLU()(conv_block2)
leakyrelu1 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block3)
conv_block4 = tf.keras.layers.MaxPool2D(strides=2,pool_size=(3,3),padding='same')(leakyrelu1)

# Convolution Block B-a
conv_block5 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(conv_block4)
conv_block6 = tf.keras.layers.BatchNormalization()(conv_block5)
conv_block7 = tf.keras.layers.ReLU()(conv_block6)
leakyrelu2 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block7)
conv_block8 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(leakyrelu2)
conv_block9 = tf.keras.layers.BatchNormalization()(conv_block8)
conv_block10 = tf.keras.layers.Add()([conv_block4,conv_block9])

# Convolution Block B-b
conv_block11 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(conv_block10)
conv_block12 = tf.keras.layers.BatchNormalization()(conv_block11)
conv_block13 = tf.keras.layers.ReLU()(conv_block12)
leakyrelu3 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block13)
conv_block14 = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same')(leakyrelu3)
conv_block15 = tf.keras.layers.BatchNormalization()(conv_block14)
conv_block16 = tf.keras.layers.Add()([conv_block10,conv_block15])

# Convolution Block C
conv_block17 = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(conv_block16)
conv_block18 = tf.keras.layers.BatchNormalization()(conv_block17)
conv_block19 = tf.keras.layers.ReLU()(conv_block18)
leakyrelu4 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block19)
conv_block20 = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(leakyrelu4)
conv_block21 = tf.keras.layers.BatchNormalization()(conv_block20)
conv_block22 = tf.keras.layers.Conv2D(128, (1,1), strides=1, padding='same')(conv_block16)
conv_block23 = tf.keras.layers.Add()([conv_block21,conv_block22])

conv_block24 = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(conv_block23)
conv_block25 = tf.keras.layers.BatchNormalization()(conv_block24)
conv_block26 = tf.keras.layers.ReLU()(conv_block25)
leakyrelu5 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block26)
conv_block27 = tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same')(leakyrelu5)
conv_block28 = tf.keras.layers.BatchNormalization()(conv_block27)
conv_block29 = tf.keras.layers.Add()([conv_block23,conv_block28])

# Convolution Block D
conv_block30 = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(conv_block29)
conv_block31 = tf.keras.layers.BatchNormalization()(conv_block30)
conv_block32 = tf.keras.layers.ReLU()(conv_block31)
leakyrelu6 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block32)
conv_block33 = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(leakyrelu6)
conv_block34 = tf.keras.layers.BatchNormalization()(conv_block33)
conv_block35 = tf.keras.layers.Conv2D(256, (1,1), strides=1, padding='same')(conv_block29)
conv_block36 = tf.keras.layers.Add()([conv_block34,conv_block35])

conv_block37 = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(conv_block36)
conv_block38 = tf.keras.layers.BatchNormalization()(conv_block37)
conv_block39 = tf.keras.layers.ReLU()(conv_block38)
leakyrelu7 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block39)
conv_block40 = tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same')(leakyrelu7)
conv_block41 = tf.keras.layers.BatchNormalization()(conv_block40)
conv_block42 = tf.keras.layers.Add()([conv_block36,conv_block41])

# Convolution Block E
conv_block43 = tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same')(conv_block42)
conv_block44 = tf.keras.layers.BatchNormalization()(conv_block43)
conv_block45 = tf.keras.layers.ReLU()(conv_block44)
leakyrelu8 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block45)
conv_block46 = tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same')(leakyrelu8)
conv_block47 = tf.keras.layers.BatchNormalization()(conv_block46)
conv_block48 = tf.keras.layers.Conv2D(512, (1,1), strides=1, padding='same')(conv_block42)
conv_block49 = tf.keras.layers.Add()([conv_block47,conv_block48])

conv_block50 = tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same')(conv_block49)  
conv_block51 = tf.keras.layers.BatchNormalization()(conv_block50)
conv_block52 = tf.keras.layers.ReLU()(conv_block51)
leakyrelu9 = tf.keras.layers.LeakyReLU(alpha=0.02)(conv_block52)
conv_block53 = tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same')(leakyrelu9)
conv_block54 = tf.keras.layers.BatchNormalization()(conv_block53)
conv_block55 = tf.keras.layers.Add()([conv_block49,conv_block54])

#Final Layer
conv_block56 = tf.keras.layers.GlobalAveragePooling2D()(conv_block55)
output1 = tf.keras.layers.Dense(2, activation='softmax')(conv_block56)

model = tf.keras.Model(input1, output1)


model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['acc'])
model.fit(train_ds, validation_data=val_ds, batch_size=64,epochs=50)