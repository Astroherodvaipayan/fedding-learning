from flask import Flask, request, jsonify
import threading
import warnings
import flwr as fl
import numpy as np
import utils
import pickle

app = Flask(__name__)

# Implementing logistic regression from scratch
class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights)
            y_predicted = self.sigmoid(linear_model)
            gradient = np.dot(X.T, (y_predicted - y)) / n_samples
            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights)
        return self.sigmoid(linear_model)

    def predict(self, X):
        return self.predict_proba(X) >= 0.5

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

# Flower client class
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = LogisticRegression(learning_rate=0.01, max_iter=1000)

    def get_parameters(self, config): 
        return self.model.weights

    def fit(self, parameters, config): 
        self.model.weights = parameters
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
            filename = f"model/client3/client_3_round_{config['server_round']}_model.sav"
            pickle.dump(self.model, open(filename, 'wb'))
        return self.model.weights, len(self.X_train), {}

    def evaluate(self, parameters, config): 
        self.model.weights = parameters
        preds = self.model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, preds, labels=[1,0])
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def start_flower_client(X_train, y_train, X_test, y_test):
    flower_client = FlowerClient(X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="localhost:8080", client=flower_client)

@app.route('/train2', methods=['POST'])
def train():
    try:
        # Load data
        (X_train, y_train), (X_test, y_test) = utils.load_data(client="client3")
        
        # Take only 100 rows
        X_train = X_train[:100]
        y_train = y_train[:100]
        
        # Partition data
        partition_id = np.random.choice(10)
        (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

        # Initialize model
        global model
        model = LogisticRegression(
            learning_rate=0.01,
            max_iter=1000
        )

        # Set initial parameters
        model.weights = np.zeros(X_train.shape[1])

        # Print initial weights
        weights = model.weights.tolist()
        print("Initial Weights:", weights)
        
        # Start Flower client in a separate thread
        thread = threading.Thread(target=start_flower_client, args=(X_train, y_train, X_test, y_test))
        thread.start()
        
        return jsonify({"weights": weights})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002, debug=True)
