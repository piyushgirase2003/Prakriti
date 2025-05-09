import streamlit as st
import pandas as pd
import pickle

# Set Streamlit page config
st.set_page_config(page_title="Prakriti Classifier", page_icon="🍿", layout="centered")

# -----------------------
# Load Model and Encoders
# -----------------------
with open("prakriti_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("prakriti_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

features = [f for f in label_encoders if f != "prakriti"]
options = {feature: label_encoders[feature].classes_.tolist() for feature in features}

# -----------------------
# Remedies Dictionary
# -----------------------
remedies = {
    'Vata': """
### 🌿 Vata Dosha (Air)
**🔹 Common Issues**: Dry skin, bloating, anxiety, joint pain, insomnia  
**🔹 Balance with**: Warm, oily, and grounding foods & habits  

**✅ Home Remedies**  
- **Sesame Oil Massage** – Reduces dryness  
- **Ginger & Ajwain Tea** – Boosts digestion  
- **Warm Milk with Nutmeg** – Aids better sleep  
- **Soaked Almonds** – Nourishes nervous system  
- **Turmeric & Ghee Mix** – Reduces joint pain  
**📌 Avoid**: Cold foods, raw vegetables, excessive fasting
""",
    'Pitta': """
### 🔥 Pitta Dosha (Fire)
**🔹 Common Issues**: Acid reflux, inflammation, irritability, skin rashes  
**🔹 Balance with**: Cooling, hydrating, and calming remedies  

**✅ Home Remedies**  
- **Aloe Vera Juice** – Cools acidity  
- **Coconut Water** – Naturally hydrating  
- **Coriander & Fennel Tea** – Soothes digestion  
- **Sandalwood Paste** – Reduces rashes  
- **Cucumber & Mint Smoothie** – Cools internal heat  
**📌 Avoid**: Spicy foods, fermented foods, caffeine
""",
    'Kapha': """
### 🌍 Kapha Dosha (Earth & Water)
**🔹 Common Issues**: Weight gain, sluggish digestion, mucus buildup, lethargy  
**🔹 Balance with**: Light, warm, and stimulating foods  

**✅ Home Remedies**  
- **Honey & Warm Water** – Burns excess fat  
- **Ginger & Black Pepper Tea** – Stimulates metabolism  
- **Turmeric & Cinnamon Milk** – Boosts immunity  
- **Triphala Powder** – Detoxifies body  
- **Dry Brushing** – Improves circulation  
**📌 Avoid**: Dairy, fried foods, excessive sweets
"""
}

# -----------------------
# Streamlit App UI
# -----------------------
st.title("🌿 Ayurvedic Prakriti Classifier")
st.write("Answer the following questions to determine your **Prakriti (Body Constitution)** and get personalized **home remedies**.")

# Form for user input
with st.form("prakriti_form"):
    user_input = {}
    for feature in features:
        user_input[feature] = st.selectbox(label=feature, options=options[feature])
    submitted = st.form_submit_button("Predict Prakriti")

# -----------------------
# Prediction and Output
# -----------------------
if submitted:
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    prediction = model.predict(input_df)[0]
    prakriti_type = label_encoders['prakriti'].inverse_transform([prediction])[0]

    st.markdown(f"## 🌿 Predicted Prakriti Type: **{prakriti_type}**")
    st.markdown(remedies.get(prakriti_type, "❌ No remedies found."))
