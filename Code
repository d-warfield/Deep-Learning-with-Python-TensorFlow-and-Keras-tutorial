
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#28x28 pixel hand-written digit images in multi-dimensional array
mnist = tf.keras.datasets.mnist 


#split data into test and train
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

#create feedforward model with 'Sequential' model
model = tf.keras.models.Sequential()

#'Flatten' multi-dimensional array units and use for input layer 
model.add(tf.keras.layers.Flatten())

#Use 'Dense' layer with 128 neurons and 'ReLU' activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Add another 'Dense' layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Create output layer using 10 neurons to to classify numbers 0-9
#Use 'softmax' activation function because we want a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#define parameters to train model
model.compile(optimizer='adam', #optimizer function
              loss='sparse_categorical_crossentropy', #loss function
              metrics=['accuracy']) #metrics we want to track

#train model
model.fit(x_train, y_train, epochs=3)

#check to see if model overfit
val_loss, val_acc = model.evaluate(x_test, y_test)
print('Validation Loss:{}\nValidation Accuracy:{}'.format(val_loss, val_acc))

#make predictions
predictions = model.predict([x_test])

#view model predictions for an individual indice
print(np.argmax(predictions[13]))

#view test image to check prediction
plt.imshow(x_test[13])
plt.show()



