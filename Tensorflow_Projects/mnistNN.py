import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train/255
x_test = x_test/255

y_onehot_train = tf.one_hot(y_train,10)
y_onehot_test = tf.one_hot(y_test,10)

model = tf.keras.models.Sequential([
    layers.Input(x_train.shape[1:]),
    layers.Flatten(),
    layers.Dense(100, activation = 'relu'),
    layers.Dense(100, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


data = model.fit(x_train,y_onehot_train,epochs=5,batch_size=128,validation_data=(x_test,y_onehot_test))
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
predictions = np.argmax(predictions,axis=1)
for i in range(3):
    print('prediction:', predictions[i])
    plt.imshow(x_test[i],cmap='Greys')
    plt.show()





