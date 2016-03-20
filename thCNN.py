import time

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from DataModel import CData

# Set hyperparameters
inshape = (1, 28, 28)
fshape = (5, 1, 3, 3)
outputs = 10
batch_size = 10
eta = 0.15
lmbd = 5.0
epochs = 30

# Pull MNIST data                      Yes, I know it's ugly
data = CData("mnist.pkl.gz", cross_val=1-0.8571428571428571)

# Calculate weight decay rate
l2term = 1 - ((lmbd*eta)/data.N)

# Define theano Tensors
X = T.tensor4("X") # Inputs
Y = T.matrix("Y")  # Targets
m = T.scalar("m", dtype='int32') # Batch size

# Define update-able shared variables
F = theano.shared(np.random.randn(*fshape))   # Convolutional filter
W = theano.shared(np.random.randn(5*13*13, 10)) # FC Layer weight matrix

# Define the activation of layers
A1 = T.nnet.sigmoid(conv.conv2d(X, F))  # Convolution
# Max pooling
A2 = T.reshape(downsample.max_pool_2d(A1, (2, 2), ignore_border=True), (m, 5*13*13))
A3 = T.nnet.softmax(A2.dot(W))

# Definition of cost function
# -(Y * T.log(A3) - (1 - Y) * T.log(1 - A3)).sum()
cost = T.nnet.categorical_crossentropy(A3, Y).sum()  # Cross-entropy cost

# Define prediction symbolically
prediction = T.argmax(A3, axis=1)

# Define the update rules for the filter and weight tensors
update_F = l2term * F - (eta / m) * T.grad(cost, F)
update_W = l2term * W - (eta / m) * T.grad(cost, W)

# Compile methods
train = theano.function(inputs=[X, Y, m], updates=[(F, update_F), (W, update_W)])
predict = theano.function(inputs=[X, Y, m], outputs=[cost, prediction])

log = open("logCNN.txt", "w")

start = time.time()

# Train the net
for i in range(epochs):
    for batch in data.batchgen(10):
        bsize = batch[0].shape[0]
        train(batch[0], batch[1], bsize)
    testtable = data.table(data="testing")
    costval, preds = predict(testtable[0], testtable[1], 10000)
    predrate = np.sum(np.equal(preds, data.dummycode("testing"))) / len(preds)
    log.write("Epoch:\t{}\tCost:\t{}\tAccuracy:\t{}\n"
              .format(i+1, costval, predrate))
    print("Epoch {} / {} done. Current prediction accuracy: {}".format(i+1, epochs, predrate))
log.write("Seconds elapsed: {}\n".format(time.time() - start))
log.close()
print("Run finished. Log dumped to logCNN.txt!")
print("Please send me the logs to csxeba@gmail.com")
print("Thank you for your help!")
    
