import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models

#training the model with data in the datasets library
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()

#training_images divided by 255 since pixels vary from 0-255
training_images = training_images/255

#testing_images divided by 255 since pixels vary from 0-255
testing_images = testing_images/255

#assining class names
class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

print(len(training_images))

#convolutional layers filter for features in images
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='softmax'))

#compiling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images ,training_labels ,epochs=10,validation_data=(testing_images,testing_labels))

loss,accuracy= model.evaluate(testing_images,testing_labels)
print(f"LOSS: {loss}")
print(f"ACCURACY: {accuracy}")

#feeding the images
img = cv.imread('../.venv/plane.png')
img = cv.cvtColor(img ,cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)
prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)

print(f"Prediction is {class_names[index]}")