from data_analysis import x_train, y_train, x_test, y_test, num_classes, path, plt, Label
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import plot_model
#from second_set_preprocess import train_generator, test_generator
import PIL.Image as Image
import PIL.ImageOps
import sys
import keras
import numpy as np

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# reshape data to be in form (num,row,col,chanel) 
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)


class conv_model:
    def __init__(self, path):
        self.conv = Sequential()
        self.path=path
        self.hist=''
        


    def conv_model_1(self,inputShape, num_classes):
        self.conv = Sequential()
        self.conv.add(Conv2D(5,(5,5), strides = (1, 1), padding='same', input_shape=inputShape, activation='relu'))
        self.conv.add(Conv2D(7,(3,3), strides = (2, 2), activation='relu'))
        self.conv.add(MaxPooling2D(pool_size=(3,3)))
        self.conv.add(Flatten())
        self.conv.add(Dense(50, activation='relu'))
        self.conv.add(Dense(20, activation='relu'))
        self.conv.add(Dense(num_classes, activation='softmax'))

    def conv_model_1_drop(self,inputShape, num_classes):
        self.conv = Sequential()
        self.conv.add(Conv2D(5,(5,5), strides = (1, 1), padding='same', input_shape=inputShape, activation='relu'))
        self.conv.add(Conv2D(6,(3,3), strides = (2, 2), activation='relu'))
        self.conv.add(Dropout(0.15))
        self.conv.add(MaxPooling2D(pool_size=(3,3)))
        self.conv.add(Flatten())
        self.conv.add(Dense(50, activation='relu'))
        self.conv.add(Dropout(0.1))
        self.conv.add(Dense(20, activation='relu'))
        self.conv.add(Dense(num_classes, activation='softmax'))
    
    def conv_model_2(self,inputShape, num_classes):
        self.conv = Sequential()
        self.conv.add(Conv2D(5,(5,5), strides = (1, 1), padding='same', input_shape=inputShape, activation='relu'))
        self.conv.add(Conv2D(6,(3,3), strides = (2, 2), activation='relu'))
        self.conv.add(MaxPooling2D(pool_size=(3,3)))
        self.conv.add(Conv2D(8,(3,3), strides = (1, 1), padding='same', input_shape=inputShape, activation='relu'))
        self.conv.add(Conv2D(6,(2,2), strides = (1, 1), activation='relu'))
        self.conv.add(AveragePooling2D(pool_size=(2,2)))
        self.conv.add(Flatten())
        self.conv.add(Dense(50, activation='relu'))
        self.conv.add(Dense(20, activation='relu'))
        self.conv.add(Dense(num_classes, activation='softmax'))
    
    def conv_model_2_drop(self,inputShape, num_classes):
        self.conv = Sequential()
        self.conv.add(Conv2D(5,(5,5), strides = (1, 1), padding='same', input_shape=inputShape, activation='relu'))
        self.conv.add(Conv2D(6,(3,3), strides = (2, 2), activation='relu'))
        self.conv.add(Dropout(0.15))
        self.conv.add(MaxPooling2D(pool_size=(3,3)))
        self.conv.add(Conv2D(8,(3,3), strides = (1, 1), padding='same', input_shape=inputShape, activation='relu'))
        self.conv.add(Conv2D(6,(2,2), strides = (1, 1), activation='relu'))
        self.conv.add(Dropout(0.1))
        self.conv.add(AveragePooling2D(pool_size=(2,2)))
        self.conv.add(Flatten())
        self.conv.add(Dense(50, activation='relu'))
        #self.conv.add(Dropout(0.08))
        self.conv.add(Dense(20, activation='relu'))
        self.conv.add(Dense(num_classes, activation='softmax'))

    def save_summary(self, fileName):
        original=sys.stdout
        sys.stdout = open(path+fileName,'w+')
        self.conv.summary()
        sys.stdout=original
    
    def save_schema(self, fileName):
        plot_model(self.conv,to_file=path+fileName)

    def model_compile(self, loss, optimizer, metrics):
        self.conv.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def model_fit_save_report(self, fileName, batch_size, epochs):
        original=sys.stdout
        sys.stdout = open(path+fileName,'w+')
        self.hist = self.conv.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                          validation_data=(x_test, y_test), shuffle=True)
        sys.stdout=original
    
    # def model_fit_gen(self, fileName, batch_size, epochs):
    #     original=sys.stdout
    #     sys.stdout = open(path+fileName,'w+')
    #     self.hist = self.conv.fit_generator(generator=train_generator, epochs=epochs,validation_data=test_generator, steps_per_epoch=train_generator.n, validation_steps=test_generator.n)
    #     sys.stdout=original
    def save_learning_curves(self, filename, title=''):
        plt.plot(self.hist.history["loss"], 'r', marker='.', label="Train Loss")
        plt.plot(self.hist.history["val_loss"], 'b', marker='.', label="Validation Loss")
        plt.legend()
        plt.grid()  
        plt.title(title)
        plt.savefig(path+filename)
        plt.close()

cm = conv_model(path)

########## model 1 no dropout
# cm.conv_model_1((28,28,1))
# cm.save_schema('cm1_schema.png')
# cm.save_summary('cm1_summary.txt')
# cm.model_compile(loss='categorical_crossentropy', optimizer=keras.optimizers.sgd(lr=0.005), metrics=['accuracy'])
# cm.model_fit_save_report('cm1_fit_report.txt', 500, 15)
# cm.save_learning_curves("cm1_learning_curves.png", "1st model without dropouts")

########## model 1 with dropout
# cm = conv_model(path)
#cm.conv_model_1_drop((28,28,1),10)
# cm.save_schema('cm1_drop_schema.png')
# cm.save_summary('cm1_drop_summary.txt')
# cm.model_compile(loss='categorical_crossentropy', optimizer=keras.optimizers.sgd(lr=0.005), metrics=['accuracy'])
# cm.model_fit_save_report('cm1_drop_fit_report.txt', 500, 50)
# cm.save_learning_curves("cm1_drop_learning_curves.png", "1st model with dropouts")
# cm.conv.save('model_1_drop')

########## model 2 no dropout
# cm.conv_model_2((28,28,1))
# cm.save_schema('cm2_schema.png')
# cm.save_summary('cm2_summary.txt')
# cm.model_compile(loss='categorical_crossentropy', optimizer=keras.optimizers.sgd(lr=0.05), metrics=['accuracy'])
# cm.model_fit_save_report('cm2_fit_report.txt', 500, 15)
# cm.save_learning_curves("cm2_learning_curves.png", '2nd model without dropouts')

######## model 2 with dropout
cm.conv_model_2_drop((28,28,1),10)
# cm.save_schema('cm2_drop_schema.png')
# cm.save_summary('cm2_drop_summary.txt')
# cm.model_compile(loss='categorical_crossentropy', optimizer=keras.optimizers.sgd(lr=0.05), metrics=['accuracy'])
# cm.model_fit_save_report('cm2_drop_fit_report.txt', 500, 50)
# cm.save_learning_curves("cm2_drop_learning_curves.png", '2nd model wit dropouts')
# cm.conv.save('model_2_drop')

cm.conv.load_weights('model_2_drop')

def model_pred(dir, img, model, title, predName):
    img = Image.open(dir+img)
    img = PIL.ImageOps.grayscale(img)
    img.save(dir + 'gray.png')
    img.thumbnail((28,28))
    img.save(dir + 'gray-scaled.png')
    im_arr = np.array(img)
    im_arr = im_arr[...,np.newaxis]
    im_arr = im_arr[np.newaxis,...]

    pred = model.predict(im_arr)

    plt.bar( Label,pred[0])
    plt.xticks(rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(dir + predName)
    plt.close()

model_pred('D:\\Studia\\SI\\Projekt\\img5\\','pullover.png',cm.conv,'Model 2 prediction','pred_model-2.png')

# print(pred[0][0])
# k=0
# for i in pred[0]:
#     print(Label[k],': ', i)
#     k+=1
########## model 2 with dropout
# cm.conv_model_2_drop((64,64,3),142)
# cm.model_compile(loss='categorical_crossentropy', optimizer=keras.optimizers.sgd(lr=0.05), metrics=['accuracy'])
# cm.model_fit_gen('cm2_drop_fit_report_d2.txt', 500, 15)
# cm.save_learning_curves("cm2_drop_learning_curves_d2.png", '2nd model wit dropouts')