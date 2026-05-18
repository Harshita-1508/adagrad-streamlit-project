import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


st.title("AdaGrad Optimizer Comparison App")

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    optimizer_choice = st.selectbox("Choose Optimizer", ["Adagrad", "Adam"])

    if optimizer_choice == "Adagrad":
        optimizer = Adagrad()
    else:
        optimizer = Adam()

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    if st.button("Train Model"):
        history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=0)

        st.write("Final Accuracy:", history.history['val_accuracy'][-1])
