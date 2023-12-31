# Hopfield Network Demonstration

* Objective: Demonstrate the pattern recognition (associative memory) capabilities of the hopfield network.

* Neural Network configuration, Test and Train Data:

```
    network = HopfieldNetwork(9)
    network.train(np.array([1,1,1,1,0,0,1,1,1]))
    network.train(np.array([1,1,1,0,1,0,0,1,0]))
    network.recall(np.array([1,0,1,0,1,0,0,1,0]))
```

* run output for basic hopfield network implementation

```
/Users/welcome/Desktop/PycharmProjects/neural_nets/.venv/bin/python /Users/welcome/Desktop/PycharmProjects/neural_nets/HopfieldNetwork.py 
(training) Pattern Weight Matrix:  [[ 1  1  1  1 -1 -1  1  1  1]
 [ 1  1  1  1 -1 -1  1  1  1]
 [ 1  1  1  1 -1 -1  1  1  1]
 [ 1  1  1  1 -1 -1  1  1  1]
 [-1 -1 -1 -1  1  1 -1 -1 -1]
 [-1 -1 -1 -1  1  1 -1 -1 -1]
 [ 1  1  1  1 -1 -1  1  1  1]
 [ 1  1  1  1 -1 -1  1  1  1]
 [ 1  1  1  1 -1 -1  1  1  1]]
(training) Pattern Weight Matrix Made Symmetric and Zeroed along Diagonal:  [[ 0  1  1  1 -1 -1  1  1  1]
 [ 1  0  1  1 -1 -1  1  1  1]
 [ 1  1  0  1 -1 -1  1  1  1]
 [ 1  1  1  0 -1 -1  1  1  1]
 [-1 -1 -1 -1  0  1 -1 -1 -1]
 [-1 -1 -1 -1  1  0 -1 -1 -1]
 [ 1  1  1  1 -1 -1  0  1  1]
 [ 1  1  1  1 -1 -1  1  0  1]
 [ 1  1  1  1 -1 -1  1  1  0]]
(training) Weight Matrices added:  [[ 0.  1.  1.  1. -1. -1.  1.  1.  1.]
 [ 1.  0.  1.  1. -1. -1.  1.  1.  1.]
 [ 1.  1.  0.  1. -1. -1.  1.  1.  1.]
 [ 1.  1.  1.  0. -1. -1.  1.  1.  1.]
 [-1. -1. -1. -1.  0.  1. -1. -1. -1.]
 [-1. -1. -1. -1.  1.  0. -1. -1. -1.]
 [ 1.  1.  1.  1. -1. -1.  0.  1.  1.]
 [ 1.  1.  1.  1. -1. -1.  1.  0.  1.]
 [ 1.  1.  1.  1. -1. -1.  1.  1.  0.]]
(training) Pattern Weight Matrix:  [[ 1  1  1 -1  1 -1 -1  1 -1]
 [ 1  1  1 -1  1 -1 -1  1 -1]
 [ 1  1  1 -1  1 -1 -1  1 -1]
 [-1 -1 -1  1 -1  1  1 -1  1]
 [ 1  1  1 -1  1 -1 -1  1 -1]
 [-1 -1 -1  1 -1  1  1 -1  1]
 [-1 -1 -1  1 -1  1  1 -1  1]
 [ 1  1  1 -1  1 -1 -1  1 -1]
 [-1 -1 -1  1 -1  1  1 -1  1]]
(training) Pattern Weight Matrix Made Symmetric and Zeroed along Diagonal:  [[ 0  1  1 -1  1 -1 -1  1 -1]
 [ 1  0  1 -1  1 -1 -1  1 -1]
 [ 1  1  0 -1  1 -1 -1  1 -1]
 [-1 -1 -1  0 -1  1  1 -1  1]
 [ 1  1  1 -1  0 -1 -1  1 -1]
 [-1 -1 -1  1 -1  0  1 -1  1]
 [-1 -1 -1  1 -1  1  0 -1  1]
 [ 1  1  1 -1  1 -1 -1  0 -1]
 [-1 -1 -1  1 -1  1  1 -1  0]]
(training) Weight Matrices added:  [[ 0.  2.  2.  0.  0. -2.  0.  2.  0.]
 [ 2.  0.  2.  0.  0. -2.  0.  2.  0.]
 [ 2.  2.  0.  0.  0. -2.  0.  2.  0.]
 [ 0.  0.  0.  0. -2.  0.  2.  0.  2.]
 [ 0.  0.  0. -2.  0.  0. -2.  0. -2.]
 [-2. -2. -2.  0.  0.  0.  0. -2.  0.]
 [ 0.  0.  0.  2. -2.  0.  0.  0.  2.]
 [ 2.  2.  2.  0.  0. -2.  0.  0.  0.]
 [ 0.  0.  0.  2. -2.  0.  2.  0.  0.]]
Bipolar Pattern:  [ 1 -1  1 -1  1 -1 -1  1 -1]
Result of Matrix Vector Multiplication:  [ 4.  8.  4. -6.  6. -4. -6.  4. -6.]
Result of Recall:  [1 1 1 0 1 0 0 1 0]

Process finished with exit code 0

```
