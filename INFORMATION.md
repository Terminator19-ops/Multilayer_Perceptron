# MULTI-LAYER PERCEPTRON
---
## Project Overview
---
### Goal
Creating a Neural Network from scratch Using mathematics in matrix operations and understanding the mathematical foundation of Neural Network

---
### Dataset Structure 
The dataset consists of 52000 values of 28*28 pixel images with numbers written from 0 to 1 and the total 784 pixels are repesented in the dataset with the value of each pixels darkness from 0 to 255 based on the brightness of the pixel
It precisely has 52000 rows and 785 columns, each row consists of a single data entry, the first column has the label of the entry followed by 784 columns of the pixel values from 1 to 28 of the first row then from 1 to 28 of second row until the 28th pixel of the 28th row

---
### Model Architecture 
The neural network consists of a input layer of 784 features, a hidden layer of 10 features and an output layer of 10 [predicting from values from 0 to 9], each layer contains its own weigths and biases for each neuron along with each layers activation functions

---
### Loss Function (Cross Entropy)

---
### Optimazation Approac
introducing batching and increasing number of neurons

---

## Major Problems Encountered

### 1. Shape Mismatch in Forward Propagation

-Symptom:
Model crashed with shape errors in matrix multiplication.

-Root Cause:
Passing a sample of shape (784,) instead of (784,1).

-How I Diagnosed It:
Used debugger → inspected shape of c_x → saw flattening caused wrong shape.

-Fix Applied:
Reshaped inputs with:

` x = X[:, j].reshape(784, 1) `


-Concept Learned:
Neural networks require column vectors; broadcasting leads to silent bugs.


