import torch
from typing import List, Optional
from torch import Tensor



class Linear:
    """
    A linear layer in a neural network.

    Methods:
        __init__(self, in_features: int, out_features: int)
            Initialize the weight matrix and bias vector.
        _init_glorot(self, in_features: int, out_features: int) -> Tensor
            Init a weight matrix with glorot initialization. Static method.
        forward(self, x: Tensor) -> Tensor
            Compute the forward pass.
    """
    def __init__(self, in_features: int, out_features: int):
        self.weight = self._init_glorot(in_features, out_features)
        self.bias = torch.zeros(out_features)

        self.weight_grad: Optional[Tensor] = None
        self.bias_grad: Optional[Tensor] = None

    @staticmethod
    def _init_glorot(in_features: int, out_features: int) -> Tensor:
        """
        Initialize a weight matrix with glorot initialization.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.

        Returns:
            (Tensor): A weight matrix with glorot initialization.
        """
        b = torch.sqrt(torch.tensor([6. / (in_features + out_features)]))
        return (2 * b) * torch.rand(in_features, out_features) - b

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            (Tensor): Output tensor.
        """
        return x @ self.weight + self.bias

class Sigmoid:
    """
    A sigmoid activation function.
    
    Methods:
        __init__(self)
            Initialize the sigmoid function.
        forward(self, x: Tensor) -> Tensor
            Compute the forward pass.
        get_gradient(self, x: Tensor) -> Tensor
            Compute the gradient of the sigmoid function.     
    """
    def __init__(self):
        self.func = lambda x: 1 / (1 + torch.exp(-x))

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            (Tensor): Output tensor.
        """
        return self.func(x)

    def get_gradient(self, x: Tensor) -> Tensor:
        """
        Compute the gradient of the sigmoid function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            (Tensor): Gradient tensor.
        """
        return self.func(x) * (1 - self.func(x))


class TanH:
    """
    A tanh activation function.
    
    Methods:
        __init__(self)
            Initialize the tanh function.
        forward(self, x: Tensor) -> Tensor
            Compute the forward pass.
        get_gradient(self, x: Tensor) -> Tensor
            Compute the gradient of the tanh function.
    """
    @staticmethod
    def forward(x: Tensor) -> Tensor:
        """
        Compute the forward pass.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            (Tensor): Output tensor.
        """
        return torch.tanh(x)

    @staticmethod
    def get_gradient(x: Tensor) -> Tensor:
        """
        Compute the gradient of the tanh function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            (Tensor): Gradient tensor.
        """
        return  1 - torch.tanh(x)**2

class MSELoss:
    """
    A mean squared error loss function.

    Methods:
        __init__(self)
            Initialize the loss function.
        forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor
            Compute the forward pass.
        get_gradient(self, y_true: Tensor, y_pred: Tensor) -> Tensor
            Compute the gradient of the loss function.
    """
    @staticmethod
    def forward(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the forward pass.
        
        Args:
            y_true (Tensor): True labels.
            y_pred (Tensor): Predicted labels.
        
        Returns:
            (Tensor): Output tensor.
        """
        return torch.mean((y_true - y_pred)**2)

    @staticmethod
    def get_gradient(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the gradient of the loss function.

        Args:
            y_true (Tensor): True labels.
            y_pred (Tensor): Predicted labels.

        Returns:
            (Tensor): Gradient tensor.
        """
        return 2 * (y_pred - y_true) / len(y_true)


# Now we bring everything together and create our neural network.
class NeuralLayer:
    """
    A neural layer in a neural network.

    Methods:
        __init__(self, in_features: int, out_features: int, activation: str)
            Initialize the linear layer and activation function.
        forward(self, x: Tensor) -> Tensor
            Compute the forward pass.
        get_weight(self) -> Tensor
            Get the weight matrix in the linear layer.
        get_bias(self) -> Tensor
            Get the weight matrix in the linear layer.
        set_weight_gradient(self, grad: Tensor) -> None
            Set a tensor as gradient for the weight in the linear layer.
        set_bias_gradient(self, grad: Tensor) -> None
            Set a tensor as gradient for the bias in the linear layer.
    """
    def __init__(self, in_features: int, out_features: int, activation: str):
        """
        Initialize the linear layer and activation function.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            activation (str): Activation function.
        """
        self.linear = Linear(in_features, out_features)
        
        if activation == 'sigmoid':
            self.act = Sigmoid()
        elif activation == 'tanh':
            self.act = TanH()
        else:
            raise ValueError('{} activation is unknown'.format(activation))

        # We save the last computation as we'll need it for the backward pass.
        self.last_input: Optional[None] = None
        self.last_zin: Optional[None] = None
        self.last_zout: Optional[None] = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            (Tensor): Output tensor.
        """
        self.last_input = x
        self.last_zin = self.linear.forward(x)
        self.last_zout = self.act.forward(self.last_zin)
        return self.last_zout

    def get_weight(self) -> Tensor:
        """
        Get the weight matrix in the linear layer.

        Returns:
            (Tensor): Weight matrix.
        """
        return self.linear.weight

    def get_bias(self) -> Tensor:
        """
        Get the bias in the linear layer.
        
        Returns:
            (Tensor): Bias.
        """
        return self.linear.bias

    def set_weight_gradient(self, grad: Tensor) -> None:
        """
        Set a tensor as gradient for the weight in the linear layer.
        
        Args:
            grad (Tensor): Gradient tensor.
        """
        self.linear.weight_grad = grad

    def set_bias_gradient(self, grad: Tensor) -> None:
        """
        Set a tensor as gradient for the bias in the linear layer.
        
        Args:
            grad (Tensor): Gradient tensor.
        """
        self.linear.bias_grad = grad


class NeuralNetwork:
    """
    A neural network.

    Methods:
        __init__(self, input_size: int, output_size: int, hidden_sizes: List[int])
            Initialize the neural network.
        forward(self, x: Tensor) -> Tensor
            Compute the forward pass.
        get_loss(self, x: Tensor, y: Tensor) -> Tensor
            Compute the loss for a dataset and given labels.
        backward(self, x: Tensor, y: Tensor) -> None
            Compute the backward pass.
        apply_gradients(self, learning_rate: float) -> None
            Update weights with the computed gradients.
        train(self, x: Tensor, y: Tensor, learning_rate: float) -> None
            Train the neural network on a dataset.
        predict(self, x: Tensor) -> Tensor
            Predict the output for a dataset.
    """
    def __init__(self, input_size, output_size, hidden_sizes: List[int]):
        """
        Initialize the neural network.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_sizes (List[int]): List of hidden layer sizes.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.layers: List[NeuralLayer] = []
        layer_sizes = [self.input_size] + self.hidden_sizes
        for i in range(1, len(layer_sizes)):
            self.layers.append(NeuralLayer(layer_sizes[i-1], layer_sizes[i], 'tanh'))
        self.layers.append(NeuralLayer(hidden_sizes[-1], self.output_size, 'sigmoid'))

        self.loss = MSELoss()

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            (Tensor): Output tensor.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Compute the loss for a dataset and given labels.
        
        Args:
            x (Tensor): Input tensor.
            y (Tensor): Labels.
            
        Returns:
            (Tensor): Loss tensor.
        """
        return self.loss.forward(self.forward(x), y)

    def backward(self, x: Tensor, y: Tensor) -> None:
        """
        Compute the backward pass.
        
        Args:
            x (Tensor): Input tensor.
            y (Tensor): Labels.
        """
        y_pred = self.forward(x)
        grad = self.loss.get_gradient(y, y_pred)
        for layer in reversed(self.layers):
            grad = grad * layer.act.get_gradient(layer.last_zin)
            layer.set_weight_gradient(layer.last_input.T @ grad)
            layer.set_bias_gradient(torch.sum(grad, dim=0))
            grad = grad @ layer.get_weight().T

        # Check if gradients have the right size.
        for i, layer in enumerate(self.layers):
            if layer.linear.weight_grad.shape != layer.linear.weight.shape \
                or layer.linear.bias_grad.shape != layer.linear.bias.shape:
                raise ValueError('Gradients in layer with index {} have a wrong shape.'
                                 .format(i))


    def apply_gradients(self, learning_rate: float) -> None:
        """Update weights with the computed gradients."""
        for layer in self.layers:
            layer.linear.weight -= learning_rate * layer.linear.weight_grad
            layer.linear.bias -= learning_rate * layer.linear.bias_grad

