# dadi_adjoint_state
## Realization of afjoin-state method with neural network (backpropagation)

We know that training a network requires three steps.

1.  Forward propagation — The first step is calculating the activation of each layer one by one, starting from the first layer phi[0], 
    until we have the activation of the last layer, phi[T] [y’].
2.  Computing the likelihood function ll(model, data) [cost function C(y’, y)]. 
3.  Backpropagation — The final step is updating the weights [and biases] of the network using the backpropagation algorithm - theta vector of parameters.

Our four step will be gradient ascent.
4.  In gradient ascent one is trying to reach the maximum of the likelihood function with respect to the parameters using the derivatives calculated in the
    back-propagation.
    
### Forward Propagation
Let xx be the input vector (matrix) to the neural network and our grid, i.e. phi[0] = phi1D(xx).
Now, we need to calculate phi[t] for every layer t in the network.

Before calculating the activation, phi[l], we will calculate an intermediate value injected_phi[l] (injected_phi - bias for the neuron k in the layer l ).
In additional we calculate derivative of functional F from vector of parameters theta.

Each element k in z[l] is just the sum of bias
for the neuron k in the layer l with the weighted sum of the activation of the previous layer, l-1.

We can calculate next_phi[t] from the following equation:

next_phi = inv(A) . injected_phi(previous_phi)

We can calculate aF/dtheta from the following equation:

F/dtheta = dA^(-1)/dtheta (previous_phi + injected_phi)

Where ‘.’ refers to the matrix multiplication operation, and + refers to the matrix addition. A - tridiagonal matrix which not dependent on theta.
