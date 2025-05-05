import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Load and train model ----
@st.cache_data
def train_model():
    crop = pd.read_csv("Crop_recommendation.csv")

    crop_dict = {
        'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
        'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
        'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
        'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
        'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
    }

    crop['crop_num'] = crop['Label'].map(crop_dict)
    crop.drop(['Label'], axis=1, inplace=True)

    X = crop.drop(['crop_num'], axis=1)
    y = crop['crop_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ms = StandardScaler()
    X_train = ms.fit_transform(X_train)
    X_test = ms.transform(X_test)

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    ypred = rfc.predict(X_test)
    acc = accuracy_score(y_test, ypred)

    return rfc, ms, acc, crop_dict

# ---- Recommendation function ----
def recommendation(rfc, ms, N, P, K, temperature, humidity, ph, rainfall):
    if not (0 <= N <= 200 and 0 <= P <= 200 and 0 <= K <= 200 and
            8 <= temperature <= 50 and 10 <= humidity <= 100 and
            3 <= ph <= 9 and 20 <= rainfall <= 300):
        return None

    features = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                            columns=['N', 'P', 'K', 'Temperature', 'Humidity', 'ph', 'Rainfall'])

    transformed_features = ms.transform(features)
    prediction = rfc.predict(transformed_features)
    return prediction[0]

# ---- Streamlit UI ----
st.set_page_config(page_title="🌱 Advanced Crop Recommendation", layout="centered")
st.title("🌾 Advanced Crop Recommendation System")

rfc, ms, acc, crop_dict = train_model()
reverse_crop_dict = {v: k.capitalize() for k, v in crop_dict.items()}

st.markdown(f"✅ **Model Accuracy:** {acc:.2f}")

st.markdown("### Enter the environmental parameters:")

N = st.number_input("Nitrogen (N)")
st.markdown("<small style='color:gray'>Valid range: 0–200</small>", unsafe_allow_html=True)

P = st.number_input("Phosphorus (P)")
st.markdown("<small style='color:gray'>Valid range: 0–200</small>", unsafe_allow_html=True)

K = st.number_input("Potassium (K)")
st.markdown("<small style='color:gray'>Valid range: 0–200</small>", unsafe_allow_html=True)

temperature = st.number_input("Temperature (°C)")
st.markdown("<small style='color:gray'>Valid range: 8–50°C</small>", unsafe_allow_html=True)

humidity = st.number_input("Humidity (%)")
st.markdown("<small style='color:gray'>Valid range: 10–100%</small>", unsafe_allow_html=True)

ph = st.number_input("pH")
st.markdown("<small style='color:gray'>Valid range: 3–9</small>", unsafe_allow_html=True)

rainfall = st.number_input("Rainfall (mm)")
st.markdown("<small style='color:gray'>Valid range: 20–300 mm</small>", unsafe_allow_html=True)

if st.button("🌿 Recommend Crop"):
    prediction = recommendation(rfc, ms, N, P, K, temperature, humidity, ph, rainfall)
    if prediction:
        crop_name = reverse_crop_dict.get(prediction, "Unknown Crop")
        st.success(f"✅ **Recommended Crop:** {crop_name}")
    else:
        st.error("🚫 Sorry, we are not able to recommend a proper crop for this environment.")

st.markdown("---")

