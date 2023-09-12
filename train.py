import os
import time
import numpy
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import average_precision_score
from sklearn.model_selection import GridSearchCV, train_test_split

from concrete.ml.sklearn import RandomForestClassifier as ConcreteRandomForestClassifier

def train(dev_folder="./dev"):
    # Download the data-sets
    if not os.path.isfile("./files/titanic.csv"):
        raise ValueError(
            "no dataset"
        )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(current_dir, "files/titanic.csv"))

    def encode_age(df):
        df.Age = df.Age.fillna(-0.5)
        bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
        categories = pd.cut(df.Age, bins, labels=False)
        df.Age = categories
        return df


    def encode_fare(df):
        df.Fare = df.Fare.fillna(-0.5)
        bins = (-1, 0, 8, 15, 31, 1000)
        categories = pd.cut(df.Fare, bins, labels=False)
        df.Fare = categories
        return df


    def encode_df(df):
        df = encode_age(df)
        df = encode_fare(df)
        sex_mapping = {"male": 0, "female": 1}
        df = df.replace({"Sex": sex_mapping})
        embark_mapping = {"S": 1, "C": 2, "Q": 3}
        df = df.replace({"Embarked": embark_mapping})
        df.Embarked = df.Embarked.fillna(0)
        df["Company"] = 0
        df.loc[(df["SibSp"] > 0), "Company"] = 1
        df.loc[(df["Parch"] > 0), "Company"] = 2
        df.loc[(df["SibSp"] > 0) & (df["Parch"] > 0), "Company"] = 3
        df = df[
            [
                "PassengerId",
                "Pclass",
                "Sex",
                "Age",
                "Fare",
                "Embarked",
                "Company",
                "Survived",
            ]
        ]
        return df


    train = encode_df(data)

    X_all = train.drop(["Survived", "PassengerId"], axis=1)
    y_all = train["Survived"]

    num_test = 0.20
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=num_test, random_state=23
    )

    # n_estimators = 50
    # max_depth = 4
    # n_bits = 6
    # n_jobs_xgb = 1
    # n_jobs_gridsearch = -1

    # A gridsearch to find the best parameters
    parameters = {
        "n_bits": [6, 8, 12, 16],
        "max_depth": [2, 4, 8, 12, 16],
        "n_estimators": [10, 30, 50, 100, 150],
        "n_jobs": [-1],
    }

 
    concrete_clf = ConcreteRandomForestClassifier()
    # concrete_clf.fit(X_train, y_train)
    # concrete_predictions = concrete_clf.predict(X_test)

    grid_search = GridSearchCV(concrete_clf, parameters, cv=3, n_jobs=1, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    # Check the accuracy of the best model
    print(f"Best score: {grid_search.best_score_}")

    # Check best hyper-parameters
    print(f"Best parameters: {grid_search.best_params_}")

    # Extract best model
    best_model = grid_search.best_estimator_
    assert isinstance(best_model, ConcreteRandomForestClassifier)   

    best_model.compile(X_train)

    # Export the final model such that we can reuse it in a client/server environment

    # Save the model to be pushed to a server later
    from concrete.ml.deployment import FHEModelDev

    fhe_api = FHEModelDev(dev_folder, best_model)
    fhe_api.save()


if __name__ == "__main__":
    train()