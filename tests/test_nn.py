import unittest
from torch import tensor
from networks.Linear.feed_forward import Linear, Sigmoid, TanH, MSELoss, NeuralLayer, NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Define a simple neural network for testing
        self.input_size = 2
        self.output_size = 1
        self.hidden_sizes = [3, 4]
        self.nn = NeuralNetwork(self.input_size, self.output_size, self.hidden_sizes)

    def test_linear_forward(self):
        linear_layer = Linear(2, 3)
        x = tensor([[1.0, 2.0]])
        output = linear_layer.forward(x)
        self.assertEqual(output.shape, (1, 3))

    def test_sigmoid_forward(self):
        sigmoid_layer = Sigmoid()
        x = tensor([[0.0, 1.0]])
        output = sigmoid_layer.forward(x)
        self.assertEqual(output.shape, (1, 2))

    def test_tanh_forward(self):
        x = tensor([[0.0, 1.0]])
        output = TanH.forward(x)
        self.assertEqual(output.shape, (1, 2))

    def test_mse_loss_forward(self):
        y_true = tensor([[1.0]])
        y_pred = tensor([[0.8]])
        loss = MSELoss.forward(y_true, y_pred)
        self.assertAlmostEqual(loss.item(), 0.04, places=2)

    def test_neural_layer_forward(self):
        layer = NeuralLayer(2, 3, 'tanh')
        x = tensor([[1.0, 2.0]])
        output = layer.forward(x)
        self.assertEqual(output.shape, (1, 3))

    def test_neural_network_forward(self):
        x = tensor([[1.0, 2.0]])
        output = self.nn.forward(x)
        self.assertEqual(output.shape, (1, 1))

if __name__ == '__main__':
    unittest.main()
