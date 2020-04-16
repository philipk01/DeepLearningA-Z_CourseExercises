# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:33:11 2019

@author: phili
"""


import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import backend
from keras.callbacks import Callback

file = 'C:\\Users\\phili\\main\\git\\DeepLearningA-Z_HandsOnCourse\\CNN_Image_Classification\\'

script_dir = os.path.dirname(file)
# script_dir = os.path.dirname(__file__)
training_set_path = os.path.join(script_dir, 'data\\training_set')
test_set_path = os.path.join(script_dir, 'data\\test_set')

# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution
input_size = (64, 64)
classifier.add(Conv2D(32, (3, 3), input_shape=(*input_size, 3), activation='relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
# classifier.add(Dropout(0.1))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


############################
#    Image Preprocessing 
############################
# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
    
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=input_size,
                                                 batch_size=batch_size,
                                                 class_mode='binary')

training_set.class_indices

test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=input_size,
                                            batch_size=batch_size,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # num images in training set
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000) # num images in test set

# Save model
model_backup_path = os.path.join(script_dir, 'cat_or_dogs_model.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)

# preprocess image
import numpy as np
from keras.preprocessing import image
img = os.path.join(script_dir, r'data\\single_prediction\\cat_or_dog_1.jpg')
test_img = image.load_img(img, target_size = input_size)
test_img = image.img_to_array(test_img) # array of 3 dims
# expand dim by 1, batch dim as required by algorithm
test_img = np.expand_dims(test_img, axis = 0)

# Load model
from keras.models import load_model
model = load_model('cat_or_dogs_model.h5')

# prediction
result = model.predict(test_img)

result[0][0]

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("The model predicted the image to be a", prediction)