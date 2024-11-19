import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix # type: ignore
from xgboost import XGBClassifier # type: ignore

# Load the dataset
df = pd.read_csv('./diabetes_prediction_dataset.csv')

# Map categorical values in 'smoking_history' to numerical values
df['smoking_history'] = df['smoking_history'].map({
    'No Info': 0,
    'never': 0,
    'former': 1,
    'current': 1,
    'not current': 1,
    'ever': 1
})

# Map categorical values in 'gender' to numerical values
df['gender'] = df['gender'].map({
    'Female': 0,
    'Other': 0,
    'Male': 1
})

# Drop rows with any missing values after mapping
df = df.dropna()

# Separate features and target variable
X = df.drop(columns=["diabetes"])  # Features
Y = df["diabetes"]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=55)

# Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, Y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate Logistic Regression
print("\nLogistic Regression:")
print("Training Accuracy:", logistic_model.score(X_train, Y_train))
print("Testing Accuracy:", logistic_model.score(X_test, Y_test))



# XGBoost Model
xgboost_model = XGBClassifier(eval_metric='logloss')  # Removed use_label_encoder
xgboost_model.fit(X_train, Y_train)
y_pred_xgboost = xgboost_model.predict(X_test)

# Evaluate XGBoost
print("\nXGBoost:")
print("Training Accuracy:", xgboost_model.score(X_train, Y_train))
print("Testing Accuracy:", xgboost_model.score(X_test, Y_test))

