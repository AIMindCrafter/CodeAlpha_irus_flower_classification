from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load model and scaler files saved from the notebook
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'final_iris_classifier_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler_used_for_model.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "Model or scaler file not found. Make sure 'final_iris_classifier_model.pkl' and 'scaler_used_for_model.pkl' are in the project directory."
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Mapping for label encoding used in the notebook
SPECIES_MAP = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Accept JSON or form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    try:
        sl = float(data.get('sepal_length'))
        sw = float(data.get('sepal_width'))
        pl = float(data.get('petal_length'))
        pw = float(data.get('petal_width'))
    except Exception as e:
        return jsonify({'error': 'Invalid input. Provide sepal_length, sepal_width, petal_length, petal_width.'}), 400

    features = np.array([[sl, sw, pl, pw]])
    features_scaled = scaler.transform(features)

    pred = model.predict(features_scaled)[0]
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled).tolist()[0]

    response = {
        'prediction': int(pred),
        'label': SPECIES_MAP.get(int(pred), str(pred)),
        'probabilities': probabilities,
    }

    return jsonify(response)


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    # Use 0.0.0.0 for listening on all interfaces; set debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True)
