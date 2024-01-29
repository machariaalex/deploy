import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Function to load datasets
def load_datasets():
    balanced_train = pd.read_csv('balanced_train.csv')
    test_data = pd.read_csv('test_data.csv')
    return balanced_train, test_data

# Function to train models
def train_models(data):
    models = [
        LogisticRegression(max_iter=1000),
        GaussianNB(),
        KNeighborsClassifier(),
        DecisionTreeClassifier(),
        BaggingClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        RandomForestClassifier()
    ]
    for model in models:
        model.fit(data.drop(['CATEGORY'], axis=1), data['CATEGORY'])
    return models

# Function to get user input
def get_user_input(features):
    user_input = {}
    for feature in features:
        user_input[feature] = st.sidebar.slider(
            f'Select {feature}',
            float(balanced_train[feature].min()),
            float(balanced_train[feature].max())
        )
    return pd.DataFrame([user_input])

# Function to scale data using MinMaxScaler
def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# Load datasets
balanced_train, test_data = load_datasets()

# Train models
models = train_models(balanced_train)

# Streamlit App
st.title("Model Deployment with Streamlit")

# User input for features
st.sidebar.header('Input Features')
user_input_df = get_user_input(balanced_train.drop(['CATEGORY'], axis=1).columns)

# Display user input
st.write("User Input:")
st.write(user_input_df)

# Predict the category using the selected model
selected_model_index = st.sidebar.selectbox("Select a model", range(len(models)))
selected_model = models[selected_model_index]

prediction = selected_model.predict(user_input_df)
st.write(f"Predicted Category: {prediction[0]}")

# Display model evaluation results
st.header("Model Evaluation Results")
for model in models:
    accuracy = cross_val_score(model, balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'], cv=5)
    st.write(f"Accuracy of {model.__class__.__name__}: {accuracy.mean()}")

# Display predictions and probabilities
scaled_test_data = scale_data(test_data.drop(['CATEGORY'], axis=1))
predictions = selected_model.predict(scaled_test_data)
probabilities = selected_model.predict_proba(scaled_test_data)
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(dosifier_predictions_final)
