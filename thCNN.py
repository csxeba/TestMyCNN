import time

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

from csxnet.datamodel import CData

floatX = theano.config.floatX

print("Theano configs:")
print("floatX: {}".format(floatX))
print("device: {}".format(theano.config.device))

# Set hyperparameters
inshape = (1, 28, 28)
fshape = (5, 1, 3, 3)
outputs = 10
batch_size = 10
lmbd = np.array([5.0]).astype(floatX)
epochs = 30

# Pull MNIST data                      Yes, I know it's ugly
data = CData("mnist.pkl.gz", cross_val=1-0.8571428571428571)
data.standardize()

# Define theano Tensors
X = T.tensor4("X", dtype=floatX)  # Inputs
Y = T.matrix("Y", dtype=floatX)  # Targets
m = T.scalar("m", dtype='int32')  # Batch size
eta = T.scalar("eta", dtype=floatX)

# Define update-able shared variables
F = theano.shared(np.random.randn(*fshape).astype(floatX), name="Filters")
W = theano.shared(np.random.randn(5*13*13, 10).astype(floatX), name="FCWeights")

m_real = m.astype(floatX)

# Define the activation of layers
A1 = T.tanh(conv.conv2d(X, F))  # Convolution
# Max pooling
A2 = T.reshape(downsample.max_pool_2d(A1, (2, 2), ignore_border=True), (m, 5*13*13))
A3 = T.tanh(A2.dot(W))

# l2 = T.sum((F*F).sum() + (W*W).sum())
# l2 *= lmbd / (data.N * 2)

cost = T.nnet.categorical_crossentropy(A3, Y).sum()
# cost += l2
# Define prediction symbolically
prediction = T.argmax(A3, axis=1)

# Define the update rules for the filter and weight tensors
update_F = F - (eta / m_real) * T.grad(cost, F)
update_W = W - (eta / m_real) * T.grad(cost, W)

# Compile methods
train = theano.function(inputs=[X, Y, m, eta], updates=[(F, update_F), (W, update_W)],
                        allow_input_downcast=True)
predict = theano.function(inputs=[X, Y, m], outputs=[cost, prediction],
                          allow_input_downcast=True)


start = time.time()

# Train the net
for i in range(epochs):
    for batch in data.batchgen(10):
        bsize = batch[0].shape[0]
        train(batch[0], batch[1], bsize, 0.15)
    testtable = data.table(data="testing")
    costval, preds = predict(testtable[0], testtable[1], 10000)
    predrate = np.sum(np.equal(preds, data.dummycode("testing"))) / len(preds)
    log.write("Epoch:\t{}\tCost:\t{}\tAccuracy:\t{}\n"
              .format(i+1, costval, predrate))
    print("Epoch {} / {} done. Current prediction accuracy: {}".format(i+1, epochs, predrate))
log = open("logCNN.txt", "w")
log.write("Seconds elapsed: {}\n".format(time.time() - start))
log.close()
print("Run finished. Log dumped to logCNN.txt!")
print("Please send me the logs to csxeba@gmail.com")
print("Thank you for your help!")
    
