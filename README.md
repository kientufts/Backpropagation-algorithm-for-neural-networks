# Backpropagation-algorithm-for-neural-networks

## Data

We have two datasets. The [first](https://github.com/kientufts/Backpropagation-algorithm-for-neural-networks/blob/master/838.arff) captures the 8-3-8 network example data from [1]. The [second](https://github.com/kientufts/Backpropagation-algorithm-for-neural-networks/blob/master/optdigits_train.arff) is a digit recognition dataset (more info [here](https://archive.ics.uci.edu/ml/machine-learning-databases/)). For this project we have already split the data into [train](https://github.com/kientufts/Backpropagation-algorithm-for-neural-networks/blob/master/optdigits_train.arff) and [test](https://github.com/kientufts/Backpropagation-algorithm-for-neural-networks/blob/master/optdigits_test.arff) portions.

## Network structure and the backpropagation algorithm

In this prokect we apply a neural network to classification problems with more than two labels. To implement this we use multiple output units with “one hot” coding. For example, for the digit dataset, since there are 10 possible labels we will have 10 output units. The encoding on label=2 is given by assigning 0100000000 to the corresponding output units. Our code should work on any arff dataset with numerical features. Therefore it should read the dataset to figure out the number of labels and hence the corresponding number of output units to be used.

During training we use the binary labels as the required outputs of the corresponding units. During testing each output unit calculates a score for the example. This is given by the value x_i in the forward pass of the backpropagation algorithm in the slides. We then predict the label which has the highest score. We will work with networks with d (depth) hidden layers each of width w. As in the slides there is an edge (and weight) between nodes in consecutive layers. The first hidden layer is connected to all inputs (features in the examples) and the output layer is connected to the last hidden layer. In the case where d = 0 the output layer is directly connected to the input layer, i.e., there are no hidden nodes. In this case each output node is a “perceptron-like” unit - it calculates a linear threshold decision boundary over the inputs, but it differs from perceptron in that it uses the sigmoid function and has a different update rule. In this way, by varying d and w we can explore the power of large networks over single linear units, and the ability of our algorithm to successfully learn such networks.

In addition for reporting we record the number of mistakes during each training iteration and evaluate the network on the test set after each iteration so that the overall code has this structure as in the below pseudocode. The weights should be initialized with independent random numbers uniformly sampled in the range [−0.1, 0.1]. Please make sure to seed the random number generator so that your results are reproducible. The learning rate should be set to η = 0.1

## References

[1]  [Machine Learning](http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/mlbook.html), Tom M. Mitchell, McGraw-Hill, 1997.
