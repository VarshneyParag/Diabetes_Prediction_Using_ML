from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os

app = Flask(__name__)

# Load and prepare the model
csv_path = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
df = pd.read_csv(csv_path)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('index.html', result=result, features=features)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
