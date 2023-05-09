from flask import Flask, request, render_template, flash
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecret'

scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('svm_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = -1
    if request.method == 'POST':
        Age = request.form.get('Age')
        EstimatedSalary = request.form.get('EstimatedSalary')

        if Age and EstimatedSalary:
            # Convert input features to a 2D array
            input_features = np.array([[float(Age), float(EstimatedSalary)]])

            # Scale the input features using the StandardScaler
            scaled_features = scaler.transform(input_features)

            # Make predictions using the trained model
            prediction = model.predict(scaled_features)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

new_data = [[19, 12000]]
print(model.predict(new_data))