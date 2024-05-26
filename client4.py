from flask import Flask, request, jsonify
import threading
import warnings
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utils
import pickle

app = Flask(__name__)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.weights = None

    def get_parameters(self, config): 
        return utils.get_model_parameters(model)

    def fit(self, parameters, config): 
        utils.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(self.X_train, self.y_train)
            self.weights = model.coef_.tolist()  # Save the weights
            filename = f"model/client4/client_4_round_{config['server_round']}_model.sav"
            pickle.dump(model, open(filename, 'wb'))
        return utils.get_model_parameters(model), len(self.X_train), {}

    def evaluate(self, parameters, config): 
        utils.set_model_params(model, parameters)
        preds = model.predict_proba(self.X_test)
        loss = log_loss(self.y_test, preds, labels=[1,0])
        accuracy = model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}

def start_flower_client(X_train, y_train, X_test, y_test):
    flower_client = FlowerClient(X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="localhost:8080", client=flower_client)

@app.route('/train4', methods=['POST'])
def train():
    try:
        # Load data
        (X_train, y_train), (X_test, y_test) = utils.load_data(client="client4")
        
        # Take only 100 rows
        X_train = X_train[:110]
        y_train = y_train[:110]
        
        # Partition data
        partition_id = np.random.choice(10)
        (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

        # Initialize model
        global model
        model = LogisticRegression(
            solver= 'saga',
            penalty="l2",
            max_iter=10, 
            warm_start=True
        )

        # Set initial parameters
        utils.set_initial_params(model)

        # After training is completed, return the weights as JSON
        weights = model.coef_.tolist()
        
        # Print training weights
        print("Training Weights:", weights)
        
        # Start Flower client in a separate thread
        thread = threading.Thread(target=start_flower_client, args=(X_train, y_train, X_test, y_test))
        thread.start()
        
        return jsonify({"weights": weights})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
