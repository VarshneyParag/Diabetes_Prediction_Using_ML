from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load and prepare the model
df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train the model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using Random Forest instead of Logistic Regression
model.fit(X_scaled, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input data from form
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
        
        # Prepare data for prediction
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        
        # Return prediction result
        result = 'Positive' if prediction[0] == 1 else 'Negative'
        return render_template('index.html', result=result, features=features)
    
    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
