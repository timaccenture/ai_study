import numpy as np

class FCLayer:
    def __init__(self, input_size: int, output_size: int, activation: str = 'relu') -> None:

        self.activation = activation
        
        # apply he initialisation for weights
        self.weights = np.random.rand(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))

        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_biases = np.zeros((1, output_size))
        self.v_biases = np.zeros((1, output_size))

        # hyperparameters for adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def forward(self, x):
        # bad practice!!!
        self.x = x
        #sanity check
        assert self.weights.shape[0] == x.shape[1]
        z = np.dot(self.x, self.weights) + self.biases

        # relu activation, takes max elementwise in the vec
        if self.activation == "relu":
            self.z = np.maximum(0,z)
        # softmax activation, needed for output layer
        elif self.activation == "softmax":
            exp_vals = np.exp(z - np.max(z, axis=-1, keepdims=True))
            self.z = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        return self.z
    
    def backward(self, d_values, lr, t):
        # relu derivative
        if self.activation == "relu":
            d_values = d_values * (self.z > 0)
        elif self.activation == "softmax":
            for i, grad in enumerate(d_values):
                grad = grad.reshape(-1,1)
            jacobian_matrix = np.diagflat(grad) - np.dot(grad, grad.T)
            d_values[i] = np.dot(jacobian_matrix, self.z[i])

        # calc derivatives wrt weights and bias
        d_weights = np.dot(self.x.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)
        # clip grads
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)

        d_inputs = np.dot(d_values, self.weights.T)

        self.weights -= lr * d_weights
        self.biases -= lr * d_biases

        # update adam
        m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)

        m_hat_weights = m_weights / (1 - self.beta1 ** t)
        v_hat_weights = v_weights / (1 - self.beta2 ** t)
        self.weights -= lr * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)

        m_hat_biases = m_biases / (1 - self.beta1 ** t)
        v_hat_biases = v_biases / (1 - self.beta2 ** t)
        self.biases -= lr * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        return d_inputs