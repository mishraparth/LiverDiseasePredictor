from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and preprocessors
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
imputer = pickle.load(open('imputer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['age']),
            int(request.form['gender']),  # 0 for Male, 1 for Female
            float(request.form['total_bilirubin']),
            float(request.form['direct_bilirubin']),
            float(request.form['alk_phos']),
            float(request.form['alt']),
            float(request.form['ast']),
            float(request.form['proteins']),
            float(request.form['albumin']),
            float(request.form['ag_ratio'])
        ]

        input_array = np.array([data])
        scaled_input = scaler.transform(input_array)
        processed_input = imputer.transform(scaled_input)

        prediction = model.predict(processed_input)[0]
        result = "🩺 Patient likely has liver disease." if prediction == 1 else "✅ Liver is likely healthy."

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)