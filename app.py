from flask import Flask, request, jsonify  
import pandas as pd 
from sklearn.model_selection import GridSearchCV  
from sklearn.preprocessing import StandardScaler, OneHotEncoder  
from sklearn.compose import ColumnTransformer  
from sklearn.pipeline import Pipeline  # To streamline the workflow of multiple steps
from sklearn.neighbors import KNeighborsClassifier  # KNN algorithm for classification
import numpy as np  
from flask_cors import CORS 
import os  
import schedule
import time


app = Flask(__name__)


CORS(app)


# Function to train the KNN model
def train_knn_model(data):
    data['Years Since Diagnosis'] = 2024 - data['Diagnosis Year']
    
    X = data.drop(['Retinopathy Status', 'Retinopathy Probability', 'Diagnosis Year'], axis=1)
    
    y = data['Retinopathy Probability']

    y = y.round().astype(int)

    categorical_features = ['Gender', 'Diabetes Type']
    numerical_features = list(set(X.columns) - set(categorical_features))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)])

    knn_classifier = KNeighborsClassifier()

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', knn_classifier)])

    param_grid = {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors
        'classifier__weights': ['uniform', 'distance'],  # Weighting scheme
        'classifier__metric': ['euclidean', 'manhattan'],  # Distance metric
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm to use
        'classifier__leaf_size': [20, 30, 40, 50],  # Leaf size for the tree
        'classifier__p': [1, 2]  # Power parameter for the Minkowski metric
    }

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
    
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_pipeline = grid_search.best_estimator_
    
    best_pipeline.fit(X, y)

    return best_pipeline

# Function to make predictions using the trained KNN model
def predict_with_knn(model, input_data):
    # Predict the retinopathy status using the model
    predicted_labels = model.predict(input_data)
    
    # Mapping label values to retinopathy probabilities
    retinopathy_probabilities = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
    predicted_probabilities = [retinopathy_probabilities[label] for label in predicted_labels]
    
    # Return the predicted probabilities
    return predicted_probabilities


# Load the data from CSV file and train the model
data = pd.read_csv('dataset/dataset.csv')
knn_model = train_knn_model(data)  # Train the KNN model using the data
csv_file = 'dataset/dataset.csv'  # Path to the dataset CSV file

# Route for making predictions using the trained KNN model
@app.route('/predict-retinopathy', methods=['POST'])
def predict_knn():
    # Get the JSON data sent with the request
    data = request.json
    
    # Check if the required keys are present in the request data
    if data is None or not all(key in data for key in ['gender', 'diabetesType', 'systolicBP', 'diastolicBP', 'hbA1c', 'estimatedAvgGlucose', 'diagnosisYear']):
        return jsonify({'error': 'Invalid or missing data'}), 400
    
    # Create a DataFrame from the input data for prediction
    new_data = pd.DataFrame({
        'Gender': [data['gender']],
        'Diabetes Type': [data['diabetesType']],
        'Systolic BP': [data['systolicBP']],
        'Diastolic BP': [data['diastolicBP']],
        'HbA1c (mmol/mol)': [data['hbA1c']],
        'Estimated Avg Glucose (mg/dL)': [data['estimatedAvgGlucose']],
        'Diagnosis Year': [data['diagnosisYear']]
    })

    # Add a new column to calculate the years since diagnosis
    new_data['Years Since Diagnosis'] = 2024 - new_data['Diagnosis Year']

    try:
        # Predict using the trained KNN model
        prediction = predict_with_knn(knn_model, new_data)
    except Exception as e:
        # Handle any errors that may occur during prediction
        return jsonify({'error': str(e)}), 500

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

# Route for submitting new data and appending it to the dataset
@app.route('/submit-data', methods=['POST'])
def submit_data():
    # Get the JSON data from the request
    data = request.get_json()

    # Required fields for the dataset
    required_fields = ['gender', 'diabetesType', 'retinopathyStatus', 'retinopathyProbability', 'diagnosisYear', 'systolicBP', 'diastolicBP', 'hbA1c', 'estimatedAvgGlucose']
    
    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing data'}), 400
    
    # Check if the dataset CSV file already exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)  # Load existing data
    else:
        # If it doesn't exist, create a new DataFrame with the specified columns
        df = pd.DataFrame(columns=['Gender', 'Diabetes Type', 'Retinopathy Status', 'Retinopathy Probability', 'Diagnosis Year', 'Systolic BP', 'Diastolic BP', 'HbA1c (mmol/mol)', 'Estimated Avg Glucose (mg/dL)'])
    
    # Create a new entry with the received data
    new_data = {
        'Gender': data['gender'],
        'Diabetes Type': data['diabetesType'],
        'Retinopathy Status': data['retinopathyStatus'],
        'Retinopathy Probability': data['retinopathyProbability'],
        'Diagnosis Year': data['diagnosisYear'],
        'Systolic BP': data['systolicBP'],
        'Diastolic BP': data['diastolicBP'],
        'HbA1c (mmol/mol)': data['hbA1c'],
        'Estimated Avg Glucose (mg/dL)': data['estimatedAvgGlucose']
    }
    
    # Convert the new data into a DataFrame
    new_data_df = pd.DataFrame([new_data])
    
    # Append the new data to the existing DataFrame
    df = pd.concat([df, new_data_df], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(csv_file, index=False)
    
    # Return success message
    return jsonify({'message': 'Data submitted successfully'}), 200



# Daily refresh function for the model
def refresh_model():
    global knn_model
    if os.path.exists(csv_file):
        data = pd.read_csv(csv_file)
        knn_model = train_knn_model(data)
        print("Model retrained and refreshed.")

# Schedule model refresh every 24 hours
schedule.every().day.at("00:00").do(refresh_model)

# Run scheduled tasks in a separate thread
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
 
    app.run(debug=True, port=5013, host='0.0.0.0')
