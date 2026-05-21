from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Use absolute path so app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "columns.pkl"), "rb"))

@app.route('/')
def home():
    return render_template('index2.html')

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_df)[0]
    result = "Cancelled" if int(prediction) == 1 else "Not Cancelled"

    return jsonify({"prediction": int(prediction), "result": result})

if __name__ == "__main__":
    app.run(debug=True)
