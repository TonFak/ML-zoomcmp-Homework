Bank Marketing: Term Deposit Subscription Prediction

Midterm project for Machine Learning Zoomcamp 2026.

Problem description

A bank runs phone marketing campaigns to sell a term deposit to its clients. Calling every client is expensive, and most calls end without a sale (only about 11–12% of clients subscribe).

The goal of this project is to build a binary classification model that predicts whether a client will subscribe to a term deposit (y = yes/no) based on client data, the history of previous contacts and the current campaign. The bank can use the predicted probability to rank clients and call the most promising ones first, which saves call-centre time and increases the conversion rate.

The model is exposed as a web service: send client features as JSON, get back the subscription probability.

Important design decision: the feature duration (length of the last call, in seconds) is excluded from training. It is only known after the call has finished, so it cannot be used to decide whom to call. Using it would give an unrealistically high score (data leakage).

Dataset
Source: Bank Marketing dataset, UCI Machine Learning Repository (file bank-full.csv, separator ;).
~45,000 rows, 16 input features and the target y.
Features: client data (age, job, marital, education, default, balance, housing, loan), last contact of the current campaign (contact, day, month, duration), campaign history (campaign, pdays, previous, poutcome).
Target: y (yes / no), strongly imbalanced.
The data file is in this repository: bank-full.csv
Project structure
.
├── bank-full.csv         # dataset
├── notebook.ipynb        # EDA, model training and tuning, model selection
├── train.py              # trains the final model and saves model.bin
├── predict.py            # Flask web service
├── predict_test.py       # example request to the service
├── model.bin             # trained DictVectorizer + XGBoost model
├── requirements.txt      # Python dependencies
├── Dockerfile            # container for the web service
└── README.md

Exploratory data analysis

Done in the notebook:

structure of the data, data types, missing values and duplicates (no missing values; the value unknown appears in some categorical columns and is kept as a separate category);
target distribution (class imbalance: about 88% no and 12% yes);
distributions of numerical features split by target;
subscription rate for every categorical feature;
correlation matrix of numerical features;
feature importance: mutual information for categorical features and per-feature ROC-AUC for numerical ones.

Main findings:

Months (mar, dec, sep, oct) have the highest subscription rate
Pdays and previous turned out to be most important
The distribution of balance  is strongly skewed

Models and results

Data split: 60% train / 20% validation / 20% test (stratified by target, random_state=11). Categorical features are one-hot encoded with DictVectorizer.

Three model families were trained and tuned on the validation set, metric is ROC-AUC:

Model	Tuned parameters	Validation ROC-AUC
Decision tree	max_depth, min_samples_leaf	0.744
Random forest	n_estimators, max_depth, min_samples_leaf	0.803
XGBoost	eta, max_depth, min_child_weight, number of rounds	0.804

The final model is XGBoost (eta=0.1, max_depth=5, min_child_weight=10, 200 boosting rounds), retrained on train + validation.

Test ROC-AUC of the final model: 0.802

How to run
1. Environment
bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Notebook

Open notebook.ipynb and run all cells (EDA, training and model selection).

3. Train the final model
bash
python train.py

This reads bank-full.csv, trains the final model, prints the test ROC-AUC and saves model.bin.

4. Run the web service locally
bash
python predict.py

The service starts on http://localhost:9696. For production use gunicorn:

bash
gunicorn --bind 0.0.0.0:9696 predict:app
5. Run with Docker

model.bin must exist before building (run train.py first).

bash
docker build -t bank-marketing .
docker run -p 9696:9696 bank-marketing
6. Send a request
bash
python predict_test.py

or with curl:

bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42, "job": "management", "marital": "married", "education": "tertiary",
    "default": "no", "balance": 1500, "housing": "yes", "loan": "no",
    "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
  }'

Example response:

json
{"subscribe": false, "subscription_probability": 0.10111431777477264}

subscription_probability is the main output. The flag subscribe is true when the probability is above the threshold set in predict.py (THRESHOLD = 0.5). Because only about 12% of clients subscribe, probabilities are usually low, so the threshold can be lowered depending on how many clients the bank is able to call.

Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, Flask, gunicorn, Docker.