import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import RandomForestClassifier as ConcreteRandomForestClassifier

import gradio as gr

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

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

n_estimators = 50
max_depth = 4
n_bits = 6
n_jobs_xgb = 1
n_jobs_gridsearch = -1
concrete_clf = ConcreteRandomForestClassifier(
    n_bits=n_bits, n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs_xgb
)
concrete_clf.fit(X_train, y_train)
concrete_predictions = concrete_clf.predict(X_test)

def predict_survival(passenger_class, is_male, age, company, fare, embark_point):
    if passenger_class is None or embark_point is None:
        return None
    df = pd.DataFrame.from_dict(
        {
            "Pclass": [passenger_class + 1],
            "Sex": [0 if is_male else 1],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embark_point + 1],
            "Company": [
                (1 if "Sibling" in company else 0) + (2 if "Child" in company else 0)
            ]
        }
    )
    df = encode_age(df)
    df = encode_fare(df)
    pred = clf.predict_proba(df)[0]
    return {"Perishes": float(pred[0]), "Survives": float(pred[1])}

def concrete_predict_survival(passenger_class, is_male, age, company, fare, embark_point):
    if passenger_class is None or embark_point is None:
        return None
    df = pd.DataFrame.from_dict(
        {
            "Pclass": [passenger_class + 1],
            "Sex": [0 if is_male else 1],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embark_point + 1],
            "Company": [
                (1 if "Sibling" in company else 0) + (2 if "Child" in company else 0)
            ]
        }
    )
    df = encode_age(df)
    df = encode_fare(df)
    pred = concrete_clf.predict_proba(df)[0]
    return {"Perishes": float(pred[0]), "Survives": float(pred[1])}

# def key_gen_fn(user_symptoms: List[str]) -> Dict:
#     """
#     Generate keys for a given user.

#     Args:
#         user_symptoms (List[str]): The vector symptoms provided by the user.

#     Returns:
#         dict: A dictionary containing the generated keys and related information.

#     """
#     clean_directory()

#     if is_none(user_symptoms):
#         print("Error: Please submit your symptoms or select a default disease.")
#         return {
#             error_box2: gr.update(visible=True, value="⚠️ Please submit your symptoms first."),
#         }

#     # Generate a random user ID
#     user_id = np.random.randint(0, 2**32)
#     print(f"Your user ID is: {user_id}....")

#     client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
#     client.load()

#     # Creates the private and evaluation keys on the client side
#     client.generate_private_and_evaluation_keys()

#     # Get the serialized evaluation keys
#     serialized_evaluation_keys = client.get_serialized_evaluation_keys()
#     assert isinstance(serialized_evaluation_keys, bytes)

#     # Save the evaluation key
#     evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
#     with evaluation_key_path.open("wb") as f:
#         f.write(serialized_evaluation_keys)

#     serialized_evaluation_keys_shorten_hex = serialized_evaluation_keys.hex()[:INPUT_BROWSER_LIMIT]

#     return {
#         error_box2: gr.update(visible=False),
#         key_box: gr.update(visible=True, value=serialized_evaluation_keys_shorten_hex),
#         user_id_box: gr.update(visible=True, value=user_id),
#         key_len_box: gr.update(
#             visible=True, value=f"{len(serialized_evaluation_keys) / (10**6):.2f} MB"
#         ),
#     }


# demo = gr.Interface(
#     fn=concrete_predict_survival,
#     inputs = [
#         gr.Dropdown(["first", "second", "third"], type="index"),
#         "checkbox",
#         gr.Slider(0, 80, value=25),
#         gr.CheckboxGroup(["Sibling", "Child"], label="Travelling with (select all)"),
#         gr.Number(value=20),
#         gr.Radio(["S", "C", "Q"], type="index"),
#     ],
#     outputs = "label",
#     examples=[
#         ["first", True, 30, [], 50, "S"],
#         ["second", False, 40, ["Sibling", "Child"], 10, "Q"],
#         ["third", True, 30, ["Child"], 20, "S"],
#     ],
#     interpretation="default",
#     live=True,
# )

# if __name__ == "__main__":
#     demo.launch()


with gr.Blocks() as demo:
    with gr.Row():
        inp = [
                gr.Dropdown(["first", "second", "third"], type="index"), 
                gr.Checkbox(label="is_male"),
                gr.Slider(0, 80, value=25),
                gr.CheckboxGroup(["Sibling", "Child"], label="Travelling with (select all)"),
                gr.Number(value=20),
                gr.Radio(["S", "C", "Q"], type="index"),
            ]
        out = gr.Label()
    btn = gr.Button("Run")
    btn.click(fn=concrete_predict_survival, inputs=inp, outputs=out)

demo.launch()