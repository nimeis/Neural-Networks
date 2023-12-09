import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")

print(y_train)
y_reshape_train = y_train.reshape(-1,)

y_reshape_test = y_test.reshape(-1,)
x_train=x_train/255
x_test=x_test/255

model = tf.keras.models.Sequential([

    layers.Conv2D(128,(3,3),1,activation='relu',padding='same',input_shape=(32,32,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(128,(4,4),1,activation='relu',padding='same'),
    layers.MaxPooling2D(),
    layers.Conv2D(256,(3,3),1,activation='relu',padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(512,(3,3),1,activation='relu',padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(256,activation='relu'),
    layers.Dense(100,activation='sigmoid'),
])
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(y_reshape_train)
data = model.fit(x_train,y_reshape_train,epochs=10,batch_size=1000,validation_data=(x_test,y_reshape_test))

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
model.evaluate(x_test,y_reshape_test)

predictions = model.predict(x_test[:5])
print(predictions)
predictions = np.argmax(predictions,axis=1)
for i in range(3):
    print('prediction:', predictions[i])
    plt.imshow(x_test[i], cmap='Accent')
    plt.show()
