import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "bank-full.csv"
OUTPUT_FILE = "model.bin"

RANDOM_STATE = 11
# 'duration' is only known after the call, so it is a leaky feature.
EXCLUDED_FEATURES = ["duration"]

XGB_PARAMS = {
    "eta": 0.1,
    "max_depth": 10,
    "min_child_weight": 10,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}
NUM_BOOST_ROUND = 200


def load_data(path):
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=EXCLUDED_FEATURES)
    return df


def split(df):
    # Same split as in the notebook: 80% full train / 20% test
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["y"]
    )
    df_full_train = df_full_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = (df_full_train["y"] == "yes").astype(int).values
    y_test = (df_test["y"] == "yes").astype(int).values

    del df_full_train["y"]
    del df_test["y"]
    return df_full_train, df_test, y_full_train, y_test


def train(df_train, y_train):
    dicts = df_train.to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    features = list(dv.get_feature_names_out())
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return dv, model


def predict(df, dv, model):
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)
    d = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))
    return model.predict(d)


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)
    df_full_train, df_test, y_full_train, y_test = split(df)

    print("Training the final model...")
    dv, model = train(df_full_train, y_full_train)

    y_pred = predict(df_test, dv, model)
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}")

    with open(OUTPUT_FILE, "wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {OUTPUT_FILE}")
