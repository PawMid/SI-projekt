from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
path='D:\\Studia\\SI\\Projekt\\captured\\'

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
num_classes = 10
# print(dir(y_train))
# print(y_train[0].item)

Label=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

def show_data(data, target, labels, showFig, path, fileName):
    w=0
    plt.figure(0)
    for i in range(5):
        for j in range(2):
            plt.subplot2grid((2,5), (j,i))

            search=0
            while(target[search] != w):
                search+=1
            plt.imshow(data[search], cmap='Greys')
            plt.title(labels[w])
            plt.subplots_adjust(hspace=.5)
            w+=1
            
            plt.axis('off')
    if showFig==True:
        plt.show()
    else:
        plt.savefig(path+fileName)
    plt.close()

def count_calsses(classes, numClasses, labels, title, path, fileName, showFig):
    counter = np.zeros(numClasses,dtype=int)
    for i in classes:
        counter[i]+=1
    plt.bar(labels,counter)
    # plt.xticks(rotation=45)
    plt.title(title)
    plt.xticks(range(1, len(labels)+1), labels, size='small',rotation=45)
    plt.tight_layout()  
    if showFig==True:
        plt.show()
    else:
        plt.savefig(path+fileName)
    plt.close()

#show_data(x_train,y_train,Label,True,path,'calsses_example.png')
count_calsses(y_test, 10, Label, 'Count of classes in test dataset', path, 'count_test.png',False)
count_calsses(y_train, 10, Label, 'Count of classes in train dataset', path, 'count_train.png',False)


