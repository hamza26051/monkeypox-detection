import tensorflow as tf   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers

dataset= tf.keras.preprocessing.image_dataset_from_directory(
    "monkeyimages",
    shuffle=True,
    batch_size=32,
    image_size=(224,224)
)

classes=dataset.class_names
print(classes)
''''
plt.figure(figsize=(10,10))
for imagesbatch, labelbatch in dataset.take(1):
     for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(imagesbatch[i].numpy().astype('uint8'))
        plt.title(classes[labelbatch[i]])
        plt.axis("off")
        plt.show()
    '''
def getdatasets(dataset, training=0.8, validating=0.1, testing=0.1, shuffle_size=10000, shuffle=True):
    length=len(dataset)
    if shuffle:
        dataset= dataset.shuffle(shuffle_size, seed=12)

    trainingsize=int(training*length)
    validatesize=int(validating*length)

    trainingdata=dataset.take(trainingsize)
    validatingdata=dataset.skip(trainingsize).take(validatesize)
    testingdata=dataset.skip(trainingsize).skip(validatesize)

    return trainingdata,validatingdata,testingdata

trainds,validateds,testds=getdatasets(dataset)

trainds=trainds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validateds=validateds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
testds=testds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


resizing=tf.keras.Sequential([
    layers.Resizing(224,224),
    layers.Rescaling(1.0/255)
])
augmentation=tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2)
])

model=tf.keras.models.Sequential([
    resizing,
    augmentation,
    layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,224,224,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32,(3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(4,activation="softmax")
])


model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(trainds, epochs=180, batch_size=100, verbose=1, validation_data=validateds)
score= model.evaluate(testds)
''''
for testbatch, testlabel in testds.take(1):
    firstimg=testbatch[0].numpy().astype('uint8')
    firstlabel= testlabel[0].numpy()

    print("first imag")
    plt.imshow(firstimg)
    print("predicted label")
    print("Actual label", classes[firstlabel])
    prediction=model.predict(testbatch)
    print(classes[np.argmax(prediction[0])])
    '''

def predict(model, img):
    imagearray=tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    imagearray=tf.expand_dims(imagearray,0)

    prediction=model.predict(imagearray)
    predictedclass=classes[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)


    return predictedclass, confidence
plt.figure(figsize=(15,18))
for images, labels in testds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        predictedclass, confidence=predict(model, images[i].numpy())
        actualclass=classes[labels[i]]
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f'actual class{actualclass},\n predicted class {predictedclass}\n confidence{confidence}')
        plt.axis("off")
        
plt.show()

model.version=1
model.save('../monkeypox detection/models/monkeypox_modelupdated.keras') 
