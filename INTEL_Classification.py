from tensorflow.keras.utils import to_categorical
import os
import PIL
import PIL.Image as Image
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import glob
from sklearn.utils import shuffle


#pd.set_option('display.max_rows', None)

train_FileName = "archive/seg_train/seg_train/"
train_path = os.path.abspath("./"+train_FileName)

test_FileName = "archive/seg_test/seg_test/"
test_path = os.path.abspath("./"+test_FileName)

train_folders = glob.glob(train_path+"/*")
test_folders = glob.glob(test_path+"/*")

temp_train_location_images = []#np.empty(shape=(150,150,3,),dtype="float32")
temp_train_location_labels = []#np.empty(shape=(1,),dtype="str")

temp_test_location_images = []
temp_test_location_labels = []

for location in train_folders:
    location_images = glob.glob(location+"/*.jpg")
    for location_image in location_images:
        pil_im = Image.open(location_image).convert("RGB")
        img = np.asarray(pil_im).astype(np.float32)
        if(np.shape(img) == (150,150,3)):
            temp_train_location_images.append(img)
            temp_train_location_labels.append(location[60:])

for location in test_folders:
    location_images = glob.glob(location+"/*.jpg")
    for location_image in location_images:
        pil_im = Image.open(location_image).convert("RGB")
        img = np.asarray(pil_im).astype(np.float32)
        if(np.shape(img) == (150,150,3)):
            temp_test_location_images.append(img)
            temp_test_location_labels.append(location[58:])



n1 = len(temp_train_location_images)
train_location_images = np.asarray(temp_train_location_images).astype(np.float32)
train_location_labels = np.empty(shape=(n1,),dtype = "int")

n2 = len(temp_test_location_images)
test_location_images = np.asarray(temp_test_location_images).astype(np.float32)
test_location_labels = np.empty(shape=(n2,),dtype = "int")
# 0 buildings , 1 forest, 2 glacier, 3 mountain, 4 sea, 5 street

index = 0
for image in temp_train_location_images:
    train_location_images[index] = image
    if(temp_train_location_labels[index] == "buildings"):
        train_location_labels[index] = 0
    elif(temp_train_location_labels[index] == "forest"):
        train_location_labels[index] = 1
    elif(temp_train_location_labels[index] == "glacier"):
        train_location_labels[index] = 2
    elif(temp_train_location_labels[index] == "mountain"):
        train_location_labels[index] = 3
    elif(temp_train_location_labels[index] == "sea"):
        train_location_labels[index] = 4
    elif(temp_train_location_labels[index] == "street"):
        train_location_labels[index] = 5
    index+=1


index = 0
for image in temp_test_location_images:
    test_location_images[index] = image
    if(temp_test_location_labels[index] == "buildings"):
        test_location_labels[index] = 0
    elif(temp_test_location_labels[index] == "forest"):
        test_location_labels[index] = 1
    elif(temp_test_location_labels[index] == "glacier"):
        test_location_labels[index] = 2
    elif(temp_test_location_labels[index] == "mountain"):
        test_location_labels[index] = 3
    elif(temp_test_location_labels[index] == "sea"):
        test_location_labels[index] = 4
    elif(temp_test_location_labels[index] == "street"):
        test_location_labels[index] = 5
    index+=1

train_location_labels = to_categorical(train_location_labels,num_classes = 6)
train_location_images= train_location_images/ 255.0

test_location_labels = to_categorical(test_location_labels,num_classes = 6)
test_location_images = test_location_images/ 255.0

train_location_images, train_location_labels = shuffle(train_location_images, train_location_labels)
train_location_images, train_location_labels = shuffle(train_location_images, train_location_labels)


test_location_images, test_location_labels = shuffle(test_location_images, test_location_labels)
test_location_images, test_location_labels = shuffle(test_location_images, test_location_labels)

#print(train_location_labels)
model = Sequential()

#conv block conv conv pool dropout
# 32 filters with 3x3 layout
model.add(Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (150,150,3) ) )
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.45))

#conv block conv conv pool dropout
model.add(Conv2D(64, (3,3), activation = "relu", padding = "same"))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.35))

#classifying
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(6, activation = 'softmax'))


model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate=0.0005), metrics = ['acc'])
history = model.fit(train_location_images,train_location_labels, epochs = 12, batch_size = 30,
                    validation_data= (test_location_images, test_location_labels))

model.save("MY_CNN_Location.h5")
test_loss, test_acc = model.evaluate(test_location_images, test_location_labels)
print(test_loss, test_acc)






#for car_type in train_folders:
#    car_images = glob.glob(car_type+"/*.jpg")
#    for car_image in car_images:
#        with open(car_image,"rb") as file:
#            train_car_image_sets.append(file)
#        train_labels.append(car_type[64:]) #dependent on system path

#for car_type in test_folders:
#    car_images = glob.glob(car_type+"/*.jpg")
#    for car_image in car_images:
#        pil_im = Image.open(car_image).convert("RGB")
#        img = np.asarray(pil_im,dtype = "float")
        #img.tolist() # removes dtype notation
#        test_car_image_sets.append(img)
#        test_labels.append(car_type[63:]) # dependent on system path



#print(car_image_sets)
#test_car_image_sets = test_car_image_sets/ 255.0
#print(test_car_image_sets[0])







#count = 0
#for car_type in train_folders: #iterates through car types Acura~Volvo
#    car_images = glob.glob(car_type+"/*.jpg") #array of images of car_type

#    for car_image in car_images:
#        if(count == 10):
#            break
#        with open(car_image ,"rb") as file:
#            img = Image.open(file)
#            img.show()
#        count+=1
#    break


#data_dir = tf.keras.utils.get_file(FileName, path)
