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

print("\nclear_test    ", clear_predict_survival({'Pclass': [1], 'Sex': [0], 'Age': [25], 'Fare': [20.0], 'Embarked': [2], 'Company': [1]}))

print("encrypted_test", concrete_predict_survival({'Pclass': [1], 'Sex': [0], 'Age': [25], 'Fare': [20.0], 'Embarked': [2], 'Company': [1]}),"\n")


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
    # quant_user_symptoms = client.model.quantize_input(user_inputs)
    
    user_inputs_df = pd.DataFrame.from_dict(user_inputs)
    user_inputs_df = encode_age(user_inputs_df)
    user_inputs_df = encode_fare(user_inputs_df)

    print("user_inputs to be encrypted =\n", user_inputs_df)
    print("user_inputs to be encrypted =\n", user_inputs_df.to_numpy())
    print("user_inputs to be encrypted =\n", user_inputs_df.to_numpy())
    
    encrypted_quantized_user_inputs = client.quantize_encrypt_serialize(user_inputs_df.to_numpy())

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
        enc_dict_box: gr.update(visible=True, value=encrypted_quantized_user_inputs_shorten_hex),
    }

def send_input_fn(user_id: str, user_inputs: np.ndarray) -> Dict:
    """Send the encrypted data and the evaluation key to the server.
    """

    if is_none(user_id) or is_none(user_inputs):
        return {
            error_box4: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the symptoms have been submitted and the evaluation "
                "key has been generated before sending the data to the server.",
            )
        }

    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    if not evaluation_key_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: "
            f"The key has been generated correctly - {evaluation_key_path.is_file()=}"
        )

        return {
            error_box4: gr.update(visible=True, value="⚠️ Please generate the private key first.")
        }

    if not encrypted_input_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: The data has not been encrypted "
            f"correctly on the client side - {encrypted_input_path.is_file()=}"
        )
        return {
            error_box4: gr.update(
                visible=True,
                value="⚠️ Please encrypt the data with the private key first.",
            ),
        }

    # Define the data and files to post
    data = {
        "user_id": user_id,
        "input": user_inputs,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    # Send the encrypted input and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        print(f"Sending Data: {response.ok=}")
    return {
        error_box4: gr.update(visible=False),
        srv_resp_send_data_box: "Data sent",
    }

def run_fhe_fn(user_id: str) -> Dict:
    """Send the encrypted input and the evaluation key to the server.

    Args:
        user_id (int): The current user's ID.
    """
    if is_none(user_id):
        return {
            error_box5: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the symptoms have been submitted, the evaluation "
                "key has been generated and the server received the data "
                "before processing the data.",
            ),
            fhe_execution_time_box: None,
        }

    data = {
        "user_id": user_id,
    }

    url = SERVER_URL + "run_fhe"

    with requests.post(
        url=url,
        data=data,
    ) as response:
        if not response.ok:
            return {
                error_box5: gr.update(
                    visible=True,
                    value=(
                        "⚠️ An error occurred on the Server Side. "
                        "Please check connectivity and data transmission."
                    ),
                ),
                fhe_execution_time_box: gr.update(visible=False),
            }
        else:
            time.sleep(1)
            print(f"response.ok: {response.ok}, {response.json()} - Computed")

    return {
        error_box5: gr.update(visible=False),
        fhe_execution_time_box: gr.update(visible=True, value=f"{response.json():.2f} seconds"),
    }

def send_input_fn(user_id: str, user_inputs: np.ndarray) -> Dict:
    """Send the encrypted data and the evaluation key to the server.
    """

    if is_none(user_id) or is_none(user_inputs):
        return {
            error_box4: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the symptoms have been submitted and the evaluation "
                "key has been generated before sending the data to the server.",
            )
        }

    evaluation_key_path = KEYS_DIR / f"{user_id}/evaluation_key"
    encrypted_input_path = KEYS_DIR / f"{user_id}/encrypted_input"

    if not evaluation_key_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: "
            f"The key has been generated correctly - {evaluation_key_path.is_file()=}"
        )

        return {
            error_box4: gr.update(visible=True, value="⚠️ Please generate the private key first.")
        }

    if not encrypted_input_path.is_file():
        print(
            "Error Encountered While Sending Data to the Server: The data has not been encrypted "
            f"correctly on the client side - {encrypted_input_path.is_file()=}"
        )
        return {
            error_box4: gr.update(
                visible=True,
                value="⚠️ Please encrypt the data with the private key first.",
            ),
        }

    # Define the data and files to post
    data = {
        "user_id": user_id,
        "input": user_inputs,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    # Send the encrypted input and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        print(f"Sending Data: {response.ok=}")
    return {
        error_box4: gr.update(visible=False),
        srv_resp_send_data_box: "Data sent",
    }

def get_output_fn(user_id: str, user_inputs: np.ndarray) -> Dict:
    """Retreive the encrypted data from the server.
    """

    if is_none(user_id) or is_none(user_inputs):
        return {
            error_box6: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the server has successfully processed and transmitted the data to the client.",
            )
        }

    data = {
        "user_id": user_id,
    }

    # Retrieve the encrypted output
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            print(f"Receive Data: {response.ok=}")

            encrypted_output = response.content

            # Save the encrypted output to bytes in a file as it is too large to pass through
            # regular Gradio buttons (see https://github.com/gradio-app/gradio/issues/1877)
            encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"

            with encrypted_output_path.open("wb") as f:
                f.write(encrypted_output)
    return {error_box6: gr.update(visible=False), srv_resp_retrieve_data_box: "Data received"}

def decrypt_fn(user_id: str, user_inputs: np.ndarray) -> Dict:
    """Dencrypt the data on the `Client Side`.
    Args:
        user_id (str): The current user's ID
        user_inputs (np.ndarray): The user inputs
    Returns:
        Decrypted output
    """

    if is_none(user_id) or is_none(user_inputs):
        return {
            error_box7: gr.update(
                visible=True,
                value="⚠️ Please check your connectivity \n"
                "⚠️ Ensure that the client has successfully received the data from the server.",
            )
        }

    # Get the encrypted output path
    encrypted_output_path = CLIENT_DIR / f"{user_id}_encrypted_output"

    if not encrypted_output_path.is_file():
        print("Error in decryption step: Please run the FHE execution, first.")
        return {
            error_box7: gr.update(
                visible=True,
                value="⚠️ Please ensure that: \n"
                "- the connectivity \n"
                "- the inputs have been submitted \n"
                "- the evaluation key has been generated \n"
                "- the server processed the encrypted data \n"
                "- the Client received the data from the Server before decrypting the prediction",
            ),
            decrypt_box: None,
        }

    # Load the encrypted output as bytes
    with encrypted_output_path.open("rb") as f:
        encrypted_output = f.read()

    # Retrieve the client API
    client = FHEModelClient(path_dir=DEPLOYMENT_DIR, key_dir=KEYS_DIR / f"{user_id}")
    client.load()

    # Deserialize, decrypt and post-process the encrypted output
    output = client.deserialize_decrypt_dequantize(encrypted_output)

    print("output =\n", output)

    # top3_diseases = np.argsort(output.flatten())[-3:][::-1]
    # top3_proba = output[0][top3_diseases]

    out = {"Perishes": float(output[0][0]), "Survives": float(output[0][1])}

    print("output =\n", out)

    return {
        error_box7: gr.update(visible=False),
        decrypt_box: out,
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
    gr.Markdown(
        "### Send the encrypted data to the <span style='color:grey'>Server Side</span>"
    )
    error_box4 = gr.Textbox(label="Error ❌", visible=False)

    with gr.Row().style(equal_height=False):
        with gr.Column(scale=4):
            send_input_btn = gr.Button("Send data")
        with gr.Column(scale=1):
            srv_resp_send_data_box = gr.Checkbox(label="Data Sent", show_label=False)

    send_input_btn.click(
        send_input_fn,
        inputs=[user_id_box, out],
        outputs=[error_box4, srv_resp_send_data_box],
    )

    # ------------------------- Step 3 -------------------------
    gr.Markdown("\n")
    gr.Markdown("## Step 3: Run the FHE evaluation")
    gr.Markdown("<hr />")
    gr.Markdown("<span style='color:grey'>Server Side</span>")
    gr.Markdown(
        "Once the server receives the encrypted data, it can process and compute the output without ever decrypting the data just as it would on clear data.\n\n"
    )

    run_fhe_btn = gr.Button("Run the FHE evaluation")
    error_box5 = gr.Textbox(label="Error ❌", visible=False)
    fhe_execution_time_box = gr.Textbox(label="Total FHE Execution Time:", visible=True)
    run_fhe_btn.click(
        run_fhe_fn,
        inputs=[user_id_box],
        outputs=[fhe_execution_time_box, error_box5],
    )

    # ------------------------- Step 4 -------------------------
    gr.Markdown("\n")
    gr.Markdown("## Step 4: Decrypt the data")
    gr.Markdown("<hr />")
    gr.Markdown("<span style='color:grey'>Client Side</span>")
    gr.Markdown(
        "### Get the encrypted data from the <span style='color:grey'>Server Side</span>"
    )

    error_box6 = gr.Textbox(label="Error ❌", visible=False)

    # Step 4.1: Data transmission
    with gr.Row().style(equal_height=True):
        with gr.Column(scale=4):
            get_output_btn = gr.Button("Get data")
        with gr.Column(scale=1):
            srv_resp_retrieve_data_box = gr.Checkbox(label="Data Received", show_label=False)

    get_output_btn.click(
        get_output_fn,
        inputs=[user_id_box, out],
        outputs=[srv_resp_retrieve_data_box, error_box6],
    )

    # Step 4.1: Data transmission
    gr.Markdown("### Decrypt the output")
    decrypt_btn = gr.Button("Decrypt the output using the private secret key")
    error_box7 = gr.Textbox(label="Error ❌", visible=False)
    decrypt_box = gr.Textbox(label="Decrypted Output:")

    decrypt_btn.click(
        decrypt_fn,
        inputs=[user_id_box, out],
        outputs=[decrypt_box, error_box7],
    )

    # ------------------------- End -------------------------

demo.launch()