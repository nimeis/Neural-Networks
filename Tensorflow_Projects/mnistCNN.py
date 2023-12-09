import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255
x_train = np.expand_dims(x_train,-1)

x_test = np.expand_dims(x_test,-1)
print(y_train)
y_onehot_train = tf.one_hot(y_train,10)

y_onehot_test = tf.one_hot(y_test,10)


model = tf.keras.models.Sequential([

    layers.Conv2D(16,(3,3),1,activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(16,(3,3),1,activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,(3,3),1,activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dense(10,activation='sigmoid'),
])
model.summary()
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
print(y_onehot_train)
data = model.fit(x_train,y_onehot_train,epochs=30,batch_size=128,validation_data=(x_test,y_onehot_test))

plt.plot(data.history['loss'],label ='train')
plt.plot(data.history['val_loss'],label ='val')
plt.legend()
plt.ylabel('loss')
plt.show()

plt.plot(data.history['accuracy'])
plt.plot(data.history['val_accuracy'],label ='val')
plt.legend()
plt.ylabel('accuracy')
plt.show()
print('model evaluation:')
model.evaluate(x_test,y_onehot_test)

predictions = model.predict(x_test[:5])
print(predictions)
predictions = np.argmax(predictions,axis=1)
for i in range(3):
    print('prediction:', predictions[i])
    plt.imshow(x_test[i],cmap='Greys')
    plt.show()