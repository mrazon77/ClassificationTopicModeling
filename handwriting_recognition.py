import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

#split data into testing and training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize pixels to make values between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#create neural network model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #add flatten layer, one line of 784 pixels
model.add(tf.keras.layers.Dense(128, activation='relu')) #add dense layer
model.add(tf.keras.layers.Dense(128, activation='relu')) #add dense layer
model.add(tf.keras.layers.Dense(10, activation='softmax')) #add dense layer, softmax = makes sure outputs add up to 1

#compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, epochs=3)
 
#save model
model.save('handwritten.model')

model=tf.keras.models.load_model('handwritten.model')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1