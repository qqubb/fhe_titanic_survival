import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import RandomForestClassifier as ConcreteRandomForestClassifier

import gradio as gr

from utils import (
    CLIENT_DIR,
    CURRENT_DIR,
    DEPLOYMENT_DIR,
    INPUT_BROWSER_LIMIT,
    KEYS_DIR,
    SERVER_URL,
    TARGET_COLUMNS,
    TRAINING_FILENAME,
    clean_directory,
    load_data,
    pretty_print,
)

import requests
import subprocess
import time
from typing import Dict, List, Tuple

from concrete.ml.deployment import FHEModelClient

subprocess.Popen(["uvicorn", "server:app"], cwd=CURRENT_DIR)
time.sleep(3)

current_dir = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(current_dir, "files/titanic.csv"))

def is_none(obj) -> bool:
    """
    Check if the object is None.

    Args:
        obj (any): The input to be checked.

    Returns:
        bool: True if the object is None or empty, False otherwise.
    """
    return obj is None or (obj is not None and len(obj) < 1)

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

def collect_input(passenger_class, is_male, age, company, fare, embark_point):
    if passenger_class is None or embark_point is None:
        return None
    input_dict = {
            "Pclass": [passenger_class + 1],
            "Sex": [0 if is_male else 1],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embark_point + 1],
            "Company": [
                (1 if "Sibling" in company else 0) + (2 if "Child" in company else 0)
            ]
        }
    print(input_dict)
    return input_dict

def clear_predict_survival(input_dict):
    df = pd.DataFrame.from_dict(input_dict)
    df = encode_age(df)
    df = encode_fare(df)
    pred = clf.predict_proba(df)[0]
    return {"Perishes": float(pred[0]), "Survives": float(pred[1])}

def concrete_predict_survival(input_dict):
    df = pd.DataFrame.from_dict(input_dict)
    df = encode_age(df)
    df = encode_fare(df)
    pred = concrete_clf.predict_proba(df)[0]
    return {"Perishes": float(pred[0]), "Survives": float(pred[1])}

print("\nclear_test    ", clear_predict_survival({'Pclass': [1], 'Sex': [1], 'Age': [25], 'Fare': [20.0], 'Embarked': [2], 'Company': [1]}))

print("encrypted_test", concrete_predict_survival({'Pclass': [1], 'Sex': [1], 'Age': [25], 'Fare': [20.0], 'Embarked': [2], 'Company': [1]}),"\n")


def key_gen_fn() -> Dict:
    """
    Generate keys for a given user.

    Args:
        
    Returns:
        dict: A dictionary containing the generated keys and related information.

    """
    clean_directory()

    # Generate a random user ID
    user_id = np.random.randint(0, 2**32)
    print(f"Your user ID is: {user_id}....")

    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # Creates the private and evaluation keys on the client side
    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    # Save the evaluation key
    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    with evaluation_key_path.open("wb") as f:
        f.write(serialized_evaluation_keys)

    serialized_evaluation_keys_shorten_hex = serialized_evaluation_keys.hex()[:INPUT_BROWSER_LIMIT]

    return {
        error_box2: gr.update(visible=False),
        key_box: gr.update(visible=True, value=serialized_evaluation_keys_shorten_hex),
        user_id_box: gr.update(visible=True, value=user_id),
        key_len_box: gr.update(
            visible=True, value=f"{len(serialized_evaluation_keys) / (10**6):.2f} MB"
        ),
    }

def encrypt_fn(user_inputs: np.ndarray, user_id: str) -> None:
    """
    """

    if is_none(user_id) or is_none(user_inputs):
        print("Error in encryption step: Provide your inputs and generate the evaluation keys.")
        return {
            error_box3: gr.update(
                visible=True,
                value="⚠️ Please ensure that your inputs have been submitted and "
                "that you have generated the evaluation key.",
            )
        }

    # Retrieve the client API
    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # user_inputs = np.fromstring(user_symptoms[2:-2], dtype=int, sep=".").reshape(1, -1)
    # quant_user_symptoms = client.model.quantize_input(user_symptoms)

    encrypted_quantized_user_symptoms = client.quantize_encrypt_serialize(user_inputs)
    assert isinstance(encrypted_quantized_user_inputs, bytes)
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    with encrypted_input_path.open("wb") as f:
        f.write(encrypted_quantized_user_inputs)

    encrypted_quantized_user_inputs_shorten_hex = encrypted_quantized_user_inputs.hex()[
        :INPUT_BROWSER_LIMIT
    ]

    return {
        error_box3: gr.update(visible=False),
        input_dict_box: gr.update(visible=True, value=user_inputs),
        enc_vect_box: gr.update(visible=True, value=encrypted_quantized_user_inputs_shorten_hex),
    }

with gr.Blocks() as demo:

    # Step 1.1: Provide inputs
    with gr.Row():
        inp = [
                gr.Dropdown(["first", "second", "third"], type="index"),
                gr.Checkbox(label="is_male"),
                gr.Slider(0, 80, value=25),
                gr.CheckboxGroup(["Sibling", "Child"], label="Travelling with (select all)"),
                gr.Number(value=20),
                gr.Radio(["S", "C", "Q"], type="index"),
            ]
        out = gr.JSON()
    btn = gr.Button("Run")
    btn.click(fn=collect_input, inputs=inp, outputs=out)

    # Step 2.1: Key generation

    gen_key_btn = gr.Button("Generate the evaluation key")
    error_box2 = gr.Textbox(label="Error ❌", visible=False)
    user_id_box = gr.Textbox(label="User ID:", visible=True)
    key_len_box = gr.Textbox(label="Evaluation Key Size:", visible=False)
    key_box = gr.Textbox(label="Evaluation key (truncated):", max_lines=3, visible=False)

    gen_key_btn.click(
        key_gen_fn,
        inputs=None,
        outputs=[
            key_box,
            user_id_box,
            key_len_box,
            error_box2,
        ],
    )

    # # Step 2.2: Encrypt data locally
    gr.Markdown("### Encrypt the data")
    encrypt_btn = gr.Button("Encrypt the data using the private secret key")
    error_box3 = gr.Textbox(label="Error ❌", visible=False)

    with gr.Row():
        with gr.Column():
            input_dict_box = gr.Textbox(label="input_dict_box:", max_lines=10)
        with gr.Column():
            enc_dict_box = gr.Textbox(label="input_dict_box:", max_lines=10)

    encrypt_btn.click(
        encrypt_fn,
        inputs=[out, user_id_box],
        outputs=[
            input_dict_box,
            enc_dict_box,
            error_box3,
        ],
    )
    # # Step 2.3: Send encrypted data to the server
    # gr.Markdown(
    #     "### Send the encrypted data to the <span style='color:grey'>Server Side</span>"
    # )
    # error_box4 = gr.Textbox(label="Error ❌", visible=False)

    # with gr.Row().style(equal_height=False):
    #     with gr.Column(scale=4):
    #         send_input_btn = gr.Button("Send data")
    #     with gr.Column(scale=1):
    #         srv_resp_send_data_box = gr.Checkbox(label="Data Sent", show_label=False)

    # send_input_btn.click(
    #     send_input_fn,
    #     inputs=[user_id_box, one_hot_vect],
    #     outputs=[error_box4, srv_resp_send_data_box],
    # )

    with gr.Row():
        btn = gr.Button("Run")
        btn.click(fn=concrete_predict_survival, inputs=out, outputs=gr.Label())

demo.launch()