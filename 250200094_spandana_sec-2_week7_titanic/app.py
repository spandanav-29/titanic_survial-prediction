import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)
model = joblib.load("models/titanic_best_model.pkl")

st.title("🚢 Titanic Survival Prediction")

st.markdown(
    "### Machine Learning Based Passenger Survival Analysis"
)

st.info(
    "Enter the passenger details below to estimate the "
    "probability of survival."
)

st.divider()

st.header("👤 Passenger Details")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "🎫 Passenger Class",
        [1, 2, 3],
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )
    sex = st.selectbox(
        "👤 Gender",
        ["male", "female"]
    )
    age = st.number_input(
        "🎂 Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=1.0
    )
    sibsp = st.number_input(
        "👨‍👩‍👧 Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:
    parch = st.number_input(
        "👪 Parents / Children",
        min_value=0,
        max_value=10,
        value=0
    )
    fare = st.number_input(
        "💰 Ticket Fare",
        min_value=0.0,
        max_value=600.0,
        value=30.0,
        step=1.0
    )
    embarked = st.selectbox(
        "⚓ Port of Embarkation",
        ["S", "C", "Q"],
        help="S = Southampton, C = Cherbourg, Q = Queenstown"
    )
    family_size = sibsp + parch + 1
    st.metric(
        "👨‍👩‍👧 Family Size",
        family_size
    )

st.divider()

predict_button = st.button(
    "🔮 Predict Passenger Survival",
    use_container_width=True,
    type="primary"
)

if predict_button:
    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    death_probability = probability[0]
    survival_probability = probability[1]
    st.divider()
    st.header("📊 Prediction Result")
    if prediction == 1:
        st.success(
            "🎉 The passenger is predicted to SURVIVE."
        )
    else:
        st.error(
            "⚠️ The passenger is predicted NOT TO SURVIVE."
        )
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "🟢 Survival Probability",
            f"{survival_probability:.2%}"
        )
    with col2:
        st.metric(
            "🔴 Not Survival Probability",
            f"{death_probability:.2%}"
        )

    st.subheader("📈 Survival Probability")

    st.progress(
        float(survival_probability)
    )

    st.subheader("📋 Passenger Summary")

    summary = pd.DataFrame({
        "Details": [
            "Passenger Class",
            "Gender",
            "Age",
            "Siblings / Spouses",
            "Parents / Children",
            "Family Size",
            "Fare",
            "Embarkation Port"
        ],
        "Value": [
            pclass,
            sex.title(),
            age,
            sibsp,
            parch,
            family_size,
            f"${fare:.2f}",
            embarked
        ]
    })

    st.table(summary)

st.divider()

st.caption(
    "🚢 Titanic Survival Prediction | "
    "Machine Learning + Python + Streamlit"
)

st.caption(
    "This application provides a machine-learning prediction "
    "and is intended for educational purposes."
)