from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import csv
import glob
from data_analysis import  count_calsses

size=(100,100)
path='D:\\Studia\\SI\\Projekt\\175990_396802_bundle_archive\\'
path2='D:\\Studia\\SI\\Projekt\\captured\\'
im_path=path+'images'
format='.jpg'

# file = open(path+'styles.csv','r')
# lines = []
# for i in file.readlines():
#     lines.append(i.replace(',',';',9))
# file.close()


# file = open(path+'styles.csv','w+')
# for i in lines:
#     file.write(i)
# file.close()



def append_ext(fn):
    return fn+".jpg"

traindf=pd.read_csv(path+'styles.csv',dtype=str,delimiter=';')
traindf["id"]=traindf["id"].apply(append_ext)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

labels = traindf['articleType'].unique()

print(labels)

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=im_path+'_train',
    x_col="id",
    y_col="articleType",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

test_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory=im_path+'_train',
    x_col="id",
    y_col="articleType",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

# x,y=train_generator.next()

# def i(y):
#     for i in range(len(y)):
#         if y[i] == 1:
#             return i

# print(len(y[0]))
# print(y.shape)

# def find_label(y):
#     for name, clas in train_generator.class_indices.items():  
#         if clas == i(y):
#             return (name)

# def generate_images():
#     for i in range(len(x)):
#         lab=find_label(y[i])
#         plt.imshow(x[i])
#         plt.title(lab)
#         plt.axis('off')
#         plt.tight_layout()
#         plt.savefig(path2+lab+'.png')

#generate_images()


