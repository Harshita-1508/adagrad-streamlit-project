import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

st.title("Optimizer Comparison App")

uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview:", df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X)

    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    optimizer_choice = st.selectbox(
        "Choose Optimizer",
        ["optimal", "adaptive"]
    )

    model = SGDClassifier(
        learning_rate=optimizer_choice,
        max_iter=1000
    )

    if st.button("Train Model"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Final Accuracy: {accuracy:.2f}")