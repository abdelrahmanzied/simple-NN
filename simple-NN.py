#Import
import numpy as np

#Class
class NuralNetwork:
    def __init__(self, X, y, neurons):
        self.input = X
        self.y = y
        self.W1 = np.random.rand(neurons, self.input.shape[1])
        self.W2 = np.random.rand(self.y.shape[1], self.W1.shape[0])
        self.output = np.zeros(self.y.shape)
    
    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def sigmoid_derivat(self, x):
        return x * (1.0 - x)  
    
    def ForwardProp(self):
        self.layer1 = self.sigmoid(np.dot(self.input, self.W1.T))
        self.output = self.sigmoid(np.dot(self.layer1, self.W2.T))
    
    def BackProp(self):
        d_W2 = np.dot(self.layer1.T, (2*(self.y - self.output) * self.sigmoid_derivat(self.output)))
        d_W1 = np.dot(self.input.T,
                                            (np.dot(2*(self.y - self.output) * self.sigmoid_derivat(self.output),
                                                    self.W2) * self.sigmoid_derivat(self.layer1)))
        self.W1 += d_W1.T
        self.W2 += d_W2.T
        
    def train(self, n):
        for  i in range(n):
            self.ForwardProp()
            self.BackProp()
    
    def cost(self):
        return np.mean((np.square(self.y -  self.output)))
         

#Data
X = np.array([[0,0,1],
                       [0,1,1],
                       [1,0,1],
                       [1,1,1]])

y = np.array([[0],
                       [1],
                       [1],
                       [0]])

#Model
NN = NuralNetwork(X, y, 4)
NN.train(1000)

print('Y: \n', y)
print('Output: \n', NN.output)
print('Cost:', NN.cost())