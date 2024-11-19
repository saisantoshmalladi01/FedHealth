import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class TabularModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(TabularModel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),  # Increased neurons
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),            # Increased dropout
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def load_data():
    # Load the dataset
    data = pd.read_csv("diabetes_prediction_dataset.csv")
    
    # Selecting only the relevant columns
    data = data[['gender', 'age', 'hypertension', 'heart_disease', 
                 'smoking_history', 'bmi', 'HbA1c_level', 
                 'blood_glucose_level', 'diabetes']]
    
    # Ensure there are no missing values
    data = data.dropna()
    
    # Map categorical values in 'smoking_history'
    data['smoking_history'] = data['smoking_history'].map({
        'No Info': 0,
        'never': 0,
        'former': 1,
        'current': 1,
        'not current': 1,
        'ever': 1
    })

    # Map categorical values in 'gender'
    data['gender'] = data['gender'].map({
        'Female': 0,
        'Other': 0,
        'Male': 1
    })

    # Drop rows with any remaining missing or invalid values after mapping
    data = data.dropna()
    
    # Log-transform skewed continuous features
    data['bmi'] = data['bmi'].apply(lambda x: np.log(x + 1))
    data['blood_glucose_level'] = data['blood_glucose_level'].apply(lambda x: np.log(x + 1))
    
    # Separate features and labels
    features = data[['gender', 'age', 'hypertension', 'heart_disease', 
                     'smoking_history', 'bmi', 'HbA1c_level', 
                     'blood_glucose_level']]
    labels = data['diabetes']
    
    # Standardize the features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, labels.values


def get_client_data(client_id, num_clients):
    features, labels = load_data()
    split_size = len(features) // num_clients
    start = client_id * split_size
    end = start + split_size
    client_features = features[start:end]
    client_labels = labels[start:end]
    return train_test_split(client_features, client_labels, test_size=0.2, random_state=42)
