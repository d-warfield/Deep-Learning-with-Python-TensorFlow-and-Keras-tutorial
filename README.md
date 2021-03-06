# Deep-Learning-with-Python-TensorFlow-and-Keras-tutorial
A quick introduction to deep learning with Python, TensorFlow, and Keras

<h1>Overview</h1>
<p>This is a simple tutorial for individuals interested in learning more about Deep Learning. In this tutorial, we build a deep learning neural network that is capable of predicting handwritten digits. You will learn how to do the following:</p> 
    
* **Import** training and testing data
* **Build** a neural network model
* **Train** a neural network model
* **Evaluate** the neural network's performance

<p>To begin, use the code walkthrough below.</p>

## Import Dependencies 

Import all of your **dependencies**. Dependencies are the 'tools' you can use to make your code work its magic.

```python
import tensorflow as tf #used for creating, training, and testing neural networks
import matplotlib.pyplot as plt #used for visualizing digit images
import numpy as np #linear algebra
```

## Import your dataset

Import each image of handwritten digits. For this tutorial, we use the MNIST database: a large database of handwritten digits that is commonly used for training various image processing systems.

```python
#import 28x28 pixel hand-written digit images in multi-dimensional arrays
mnist = tf.keras.datasets.mnist 
```

## Split your dataset into training and testing
Next,split your dataset into **training** and **testing** sets.

```python
#split data into test and train
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

Splitting your data enables you to evaluate your model's 'generalization' capabilities when testing it on *unseen* training data. In this particular case, our unseen training data would be handwritten digits that our model has not seen yet seen, denoted as ```x_test``` in the code.

Furthermore, after a model has been processed by using the training set, you test the model by making predictions *against* the test set. Because the data in the testing set already contains known values for the number that you want to predict, it is easy to determine whether the model's guesses are correct.

## Normalize your data
Next, you **normalize** the data.

```python
#normalize data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
```

**Why do we normalize data**
The goal of normalization is to change the values in your dataset so they have a **common scale**. By inducing a common scale, your training data is less sensitive to the **scale** of individual features. In other words, normalization eliminates the units of measurement from your data, enabling you to more easily compare data from different places.

## Create your neural network model
Next, you create a 'feed-forward' neural network. Feed-forward neural networks are artificial neural networks where the connections between each neuron do not form a cycle.

```python
#create feedforward model with 'Sequential' model
model = tf.keras.models.Sequential()
```

## Add input and hidden layers
To create the first layer of our neural network, we convert our 28x28 multi-dimensional arrays into a vector using ```.Flatten```. This method takes each pixel value of our selected image, adds it to a vector, and then uses each number in the vector as an input value for our first layer neurons.

```python
#'Flatten' multi-dimensional array units and use for input layer 
model.add(tf.keras.layers.Flatten())
```
## Add your hidden layers
```python
#Use 'Dense' layer with 128 neurons and 'ReLU' activation function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#Add another 'Dense' layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
```

## Add your final 'output' layer
Our output layer has only 10 neurons--one for each number ranging from 0-9. We use a 'softmax' activation function so that our model can determine separate probabilities for each of the ten digits. In other words, when we instruct our model to predict a digit that we know is--let's say 7--we would want our softmax activation function to return a number somewhere close to 1.

More specifically, we have 10 different classes for our numbers ranging from 0-9. Hence, the dimension of the output layer is 10. In a perfect world, if our model could predict a probability of 1.0 for a single output, and a probability of zero for the other outputs, our model would work 100% accurately all of the time.

```python
#Create output layer using 10 neurons to classify numbers 0-9
#Use 'softmax' activation function because we want a probability distribution
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
```

## Optimize and calculate loss
When we train our neural network, we caculate **loss**. Loss is the penalty for a bad prediction. 

To reduce our loss, and, in effect, make our model predict digits more accurately, we use an **optimizer** function. The purpose of optimizer function is to 'rewire' our neural network so that it learns to make better predictions on the data we feed our network in the future. Put more formally, we use an optimizer function to adjust the individual weights of each neuron to augment the neural network's ability to learn.

```python
#define parameters to train model
model.compile(optimizer='adam', #optimizer function
              loss='sparse_categorical_crossentropy', #loss function
              metrics=['accuracy']) #metrics we want to track
```

There are many different flavors of optimizer functions and loss functions. In our case, we used ```optimizer='adam'```and ```loss='sparse_categorical_crossentropy'``` for both our optimizer and loss functions. 

Lastly, we use ``` metrics=['accuracy']``` to calculate the ** accuracy** of our model. The accuracy of our model tells us how often our model predicted the correct number. Put more formally, accuracy computes the accuracy rate across all predictions, where we compare the difference of the actual labels to the predicted labels.


## Train the model 
We begin training our model by using the method ```.fit``` and feed it two arguments: 1) Our training data, denoted as ```x_train``` and each images' corresponding label, denoted as ```y_train```

We also define our number of epochs as ```epochs=3``` which means that we will do a forward pass through the full training set precisely 3 times. This allows our backpropagation algorithm to converge on a combination of weights with an acceptable level of accuracy.

```python
#train model
model.fit(x_train, y_train, epochs=3)
```

## Test model on unseen data
Next, we compute the validation loss and validation accuracy of our model. These particular metrics help us determine how well our model 'generalizes,' or, in simple terms, **performs** on unseen data. If our model performs well on unseen data, then we can be sure that our model will make extremely accurate predictions. 

```python
#check to see if model overfit
val_loss, val_acc = model.evaluate(x_test, y_test)
print('Validation Loss:{}\nValidation Accuracy:{}'.format(val_loss, val_acc))
```

## Make a prediction
We use our ```.predict([x_test])``` method instruct our model to make predictions for the testing data. 

``` python
#make predictions
predictions = model.predict([x_test])
```

## View our prediction

```python
#view model predictions for an individual indice
print(np.argmax(predictions[13]))

#view test image to check prediction
plt.imshow(x_test[13])
```
