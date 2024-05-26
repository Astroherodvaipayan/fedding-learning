from flask import Flask, request, jsonify
import warnings
import flwr as fl
import numpy as np
import utils
import pickle

app = Flask(_name_)

class LogisticRegression:
    def _init_(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, h, y):
        return (-1 / len(y)) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    
    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for i in range(self.num_iterations):
            # Compute the linear combination of inputs and weights
            z = np.dot(X, self.weights) + self.bias
            
            # Apply sigmoid function to get probabilities
            h = self.sigmoid(z)
            
            # Compute the gradient of the cost function
            dw = (1 / len(y)) * np.dot(X.T, (h - y))
            db = (1 / len(y)) * np.sum(h - y)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        # Compute probabilities using the learned weights and bias
        z = np.dot(X, self.weights) + self.bias
        probabilities = self.sigmoid(z)
        return probabilities

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model

@app.route('/train', methods=['POST'])
def train():
    try:
        # Load data
        (X_train, y_train), (X_test, y_test) = utils.load_data(client="client1")
        
        # Take only 100 rows
        X_train = X_train[:100]
        y_train = y_train[:100]

        # Partition data
        partition_id = np.random.choice(10)
        (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

        # Initialize model
        model = LogisticRegression(learning_rate=0.01, num_iterations=1000)

        # Train the model
        trained_model = train_model(X_train, y_train, model)

        # Save the trained model
        filename = f"model/client1/client_1_round_model.sav"
        pickle.dump(trained_model, open(filename, 'wb'))

        # After training is completed, return the weights as JSON
        weights = trained_model.weights.tolist()
        return jsonify({"weights": weights})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

if _name_ == '_main_':
    app.run(debug=True)