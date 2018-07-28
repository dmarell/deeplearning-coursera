# Project settings, Create new Python SDK, Virtual Environment
# source ../venv/bin/activate
# pip install p1 p2 ...
# Right click lab4.py, Run
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))

### CONSTANTS DEFINING THE MODEL ####
layers_dims = [12288, 20, 7, 5, 1] #  5-layer model

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    np.random.seed(1)
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
pred_train = predict(train_x, parameters)
print("Train accuracy: " + str(np.sum((pred_train == train_y)/train_x.shape[1])))

pred_test = predict(test_x, parameters)
print("Test accuracy: " + str(np.sum((pred_test == test_y)/test_x.shape[1])))
# print_mislabeled_images(classes, test_x, test_y, pred_test)

def predictClass(fname):
    image = np.array(ndimage.imread(fname, flatten=False))
    imageMatrix = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
    predResult = np.squeeze(predict(imageMatrix, parameters))
    predClass = classes[int(predResult),].decode("utf-8")

    plt.imshow(image)
    print ("y = " + str(predResult) + ", file " + fname + ", predicts a \"" + predClass +  "\" picture.")
    plt.show()

    return predClass

print("katt.jpg: ", predictClass("katt.jpg"))
print("lodjur.jpg: ", predictClass("lodjur.jpg"))
print("daniel.jpg: ", predictClass("daniel.jpg"))
