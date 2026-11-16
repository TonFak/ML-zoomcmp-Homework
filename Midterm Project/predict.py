import pickle

import xgboost as xgb
from flask import Flask, jsonify, request

MODEL_FILE = "model.bin"
THRESHOLD = 0.5

with open(MODEL_FILE, "rb") as f_in:
    dv, model = pickle.load(f_in)

FEATURE_NAMES = list(dv.get_feature_names_out())

app = Flask("bank-marketing")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()
    if not isinstance(client, dict):
        return jsonify({"error": "Send a JSON object with client features"}), 400

    X = dv.transform([client])
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    probability = float(model.predict(dmatrix)[0])

    return jsonify({
        "subscription_probability": probability,
        "subscribe": probability >= THRESHOLD,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
