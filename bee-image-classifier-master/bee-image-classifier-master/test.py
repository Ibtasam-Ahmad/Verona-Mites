import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import cv2, os

def train_model():
    train_dir= "./BeeImgData/train"
    test_dir= "./BeeImgData/test"
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(50,50), color_mode='grayscale', batch_size=20, class_mode='binary', subset='training')
    val_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(50,50), color_mode='grayscale', batch_size=20, class_mode='binary', subset='validation')
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu',input_shape=(50,50,1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=1, verbose=1, validation_data=val_generator)
    
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    model.save('varona_model.h5')
    return model

def load_model():
    model = tf.keras.models.load_model('varona_model.h5')
    return model

def Single_Image_Prediction(file, model):
    image = cv2.imread(file)
    plt.imshow(image)
    plt.show()
    image = cv2.resize(image, (50,50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = img_arr / 255.0
    np_image = np.expand_dims(img_arr, axis=0)
    pred_value = model.predict(np_image)
    
    if pred_value < 0.5:
        return 'varona'
    else:
        return 'no varona'

# Example usage
model = train_model()
model = load_model()
result = Single_Image_Prediction('./BeeImgData/test/bees/038_115.png', model)
print(result)
