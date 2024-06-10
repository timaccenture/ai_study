import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from layer import FCLayer


class NN:
    def __init__(self, input_size, output_size, hidden_size) -> None:
        self.layer1 = FCLayer(input_size, hidden_size)
        self.layer2 = FCLayer(hidden_size, hidden_size)
        self.layer3 = FCLayer(hidden_size, output_size, activation='softmax')

    def forward(self, inputs):
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)
        return output3
    
    def train(self, inputs, targets, epochs, lr, decay):

        t = 0
        epsilon = 1e-10
        losses, accuracies = [], []
        for epoch in range(epochs):

            output = self.forward(inputs)
            loss = -np.mean(targets * np.log(output + epsilon))

            predicted_labels = np.argmax(output, axis=1)
            true_labels = np.argmax(targets, axis=1)
            accuracy = np.mean(predicted_labels == true_labels)

            output_grad = 6 * (output - targets) / output.shape[0]
            t += 1
            lr_hat = lr / (1 + decay * epoch)
            grad_3 = self.layer3.backward(output_grad, lr_hat, t)
            grad_2 = self.layer2.backward(grad_3, lr_hat, t)
            grad_1 = self.layer1.backward(grad_2, lr_hat, t)

            losses.append(loss)
            accuracies.append(accuracy)

            print(f"epoch {epoch} // loss {loss} // acc {accuracy}")

        plt.plot(range(epochs), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.show()

        plt.plot(range(epochs), accuracies, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy Curve')
        plt.legend()
        plt.show()

    def test(self, inputs, targets):
        output = self.forward(inputs)

        predicted_labels = np.argmax(output, axis=1)
        true_labels = np.argmax(targets, axis=1)
        acc = np.mean(true_labels == predicted_labels)

        print(f"accuracy on the test set is {acc}")

if __name__ == "__main__":
    INPUT_SIZE = 64
    HIDDEN_SIZE = 32
    OUTPUT_SIZE = 10

    digits = load_digits()

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)

    X_train = X_train.astype("float32") / 15.0
    X_test = X_test.astype("float32") / 15.0

    nn = NN(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE)
    nn.train(X_train, y_train, 500, 0.001, 0.001)

    nn.test(X_test, y_test)

