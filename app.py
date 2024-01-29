import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Assuming you have the balanced_train and test_data available
# If not, make sure to load your datasets appropriately

#loading the datasets
balanced_train = pd.read_csv('balanced_train.csv')
test_data = pd.read_csv('test_data.csv')


# Create a list of models to fit
models = [LogisticRegression(), GaussianNB(), KNeighborsClassifier(), DecisionTreeClassifier(), BaggingClassifier(), AdaBoostClassifier(), GradientBoostingClassifier(), RandomForestClassifier()]

# Fit each model to the transformed dataset
for model in models:
    model.fit(balanced_train.drop(['CATEGORY'], axis=1), balanced_train['CATEGORY'])

# Streamlit App
st.title("Model Deployment with Streamlit")

# User input for features
st.sidebar.header('Input Features')
user_input = {}
for feature in balanced_train.drop(['CATEGORY'], axis=1).columns:
    user_input[feature] = st.sidebar.slider(f'Select {feature}', float(balanced_train[feature].min()), float(balanced_train[feature].max()))

# Create a dataframe with user input
user_input_df = pd.DataFrame([user_input])

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
predictions = selected_model.predict(test_data.drop(['CATEGORY'], axis=1))
probabilities = selected_model.predict_proba(test_data.drop(['CATEGORY'], axis=1))
dosifier_predictions = pd.DataFrame(probabilities, columns=selected_model.classes_, index=test_data.index)
dosifier_predictions_final = dosifier_predictions.groupby(level=0).mean()
st.header("Test Data Predictions and Probabilities")
st.write(dosifier_predictions_final)
