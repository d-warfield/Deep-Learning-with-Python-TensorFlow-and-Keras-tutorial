# Deep-Learning-with-Python-TensorFlow-and-Keras-tutorial
A gentle introduction to deep learning with Python, TensorFlow, and Keras

<h1>Overview</h1>
This is a simple tutorial for individuals interested in learning more about Deep Learning. In this tutorial, we build a deep learning model that can predict handwritten digits. You will learn how to do the following:


* Import training and testing data
* Build a neural network model
* Train a neural network model
* Evaluation the neural network's performance


<h1>Import Dependencies</h1>

Import all of your **dependencies**

``` python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
```

<h1>Import Dataset</h1>

Import handwritten images. The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems.
``` python
#28x28 pixel hand-written digit images in multi-dimensional array
mnist = tf.keras.datasets.mnist 
```

<h1>Split your dataset into training and testing</h1>
Split your dataset into **training** and **testing** sets.
``` python
#split data into test and train
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```
Splitting your data enables you to evaluate your model's 'generalization' capabilities when testing it on *unseen* training data. In this particular case, our unseen training data would be handwritten digits that our model has not seen yet seen, denoted as ```x_test``` in the code.

Furthermore, after a model has been processed by using the training set, you test the model by making predictions *against* the test set. Because the data in the testing set already contains known values for the attribute that you want to predict, it is easy to determine whether the model's guesses are correct.

<h1>Normalize your data</h1>
Next, you **normalize** the data. 
```
#normalize data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
```
**Why do we normalize the data**
The goal of normalization is to change the values in your dataset so they have a **common scale**. By inducing a common scale, your training data less sensitive to the scale of individual features. In other words, normalization eliminates the units of measurement from your data, enabling you to more easily compare data from that different places.

<h1>Create your neural network model</h1>
Next, you create a 'feed-foward' neural network. Feed-forward neural networks are artificial neural networks where the connections between each nueron [unit] do not form a cycle.
```
#create feedforward model with 'Sequential' model
model = tf.keras.models.Sequential()
```
<h1>Add input and hidden layers</h1>
```python
#'Flatten' multi-dimensional array units and use for input layer 
model.add(tf.keras.layers.Flatten())
```
Add your hidden layers
```python
#Use 'Dense' layer with 128 neurons and 'ReLU' activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Add another 'Dense' layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
```

<h1>Add your final 'output' layer</h1>
Our output layer has only 10 neurons--one for each number ranging from 0-9. We use a 'softmax' activiation function so that our model can determine separate probabilities for each of the ten digits. When we instruct our model to make a prediction on a digit that we know is--let's say 7--we would want our softmax activation function to return a number somewhere close to 1. If, however, our model distributes  

We have 10 different classes for our numbers ranging from 0-9. Hence, the dimension of the output layer is 10. In a perfect world, if our model could predict a probability of 1.0 for a single output, and probability of zero for the other outputs, our model would work 100% of the time.

```python
#Create output layer using 10 neurons to to classify numbers 0-9
#Use 'softmax' activation function because we want a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
```

<h1>Optimize and calculate loss</h1>
When we train our neural network, we cacluate **loss**. Loss is the penalty for a bad prediction. 

To reduce our loss and, in effect, make our model predict digits more accurately, we use an **optimizer** function. The purpose of optimizer function is to 'rewire' our neural network so that it learns to make better predictions on the data we our network in the future. 

```
#define parameters to train model
model.compile(optimizer='adam', #optimizer function
              loss='sparse_categorical_crossentropy', #loss function
              metrics=['accuracy']) #metrics we want to track
```

There are many different flavors of optimizer functions and loss functions. In our case, we used ```optimizer='adam'```and ```loss='sparse_categorical_crossentropy'``` for both our optimizer and loss functions. 

Lastly, we use ``` metrics=['accuracy']``` to calculate ** accuracy** of our model. The accuracy of our model tells us how often our model predicted the right number. More formally put, accuracy computes the accuracy rate across all predictions, where we compare the difference of the actual labels and the predicted labels.


<h1>Train the model</h1>
We begin training our model by using the method ```.fit``` and feed it two arguments: 1) Our training data, denoted as ```x_train``` and each images' corresponding label, denoted as ```y_train```

We also define our number of epochs as ```epochs=3``` which means that we will do a forward pass through the full training set precisely 3 times. This allows our backpropagation algorithm to converge on a combination of weights with an acceptable level of accuracy.

```python
#train model
model.fit(x_train, y_train, epochs=3)
```
<h1>Test model on unseen data</h1>
Next, we compute the validation loss and validation accuracy of our model. These particular metrics help us determine how well our model 'generalizes,' or, in simple terms, **performs** on **unseen** data. If our model performs well on unseen data, then we can be sure that our model will make extremely accurate predictions. 

```python
#check to see if model overfit
val_loss, val_acc = model.evaluate(x_test, y_test)
print('Validation Loss:{}\nValidation Accuracy:{}'.format(val_loss, val_acc))
```

<h1>Make a prediction</h1>
We use our ```.predict([x_test])``` method instruct our model to make predictions for the testing data. 
``` python
#make predictions
predictions = model.predict([x_test])
```

<h1>View our prediction</h1>
```python
#view model predictions for an individual indice
print(np.argmax(predictions[13]))

#view test image to check prediction
plt.imshow(x_test[13])
