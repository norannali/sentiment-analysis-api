import streamlit as st
import requests
import pandas as pd

#  Configure Streamlit page
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")

#  Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextArea textarea {
            border-radius: 10px;
            border: 1px solid #ccc;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 8px 20px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

#  App Title
st.title("💬 Sentiment Analysis App")
st.write("Enter one or multiple texts to predict sentiment (Positive / Negative). For multiple texts, write each one on a new line.")

#  User Input
text_input = st.text_area("Enter text(s):", height=200, placeholder="Example:\nI love this product!\nThis was the worst experience ever.")

#  Predict button
if st.button("🔮 Predict"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        try:
            # Split user input into multiple lines
            sentences = [s.strip() for s in text_input.split("\n") if s.strip()]

            # Decide whether to call single or batch API
            if len(sentences) == 1:
                url = "http://127.0.0.1:5000/predict"
                response = requests.post(url, json={"text": sentences[0]})
            else:
                url = "http://127.0.0.1:5000/predict/batch"
                response = requests.post(url, json={"texts": sentences})

            # Process response
            if response.status_code == 200:
                result = response.json()

                if "predictions" in result:  # Batch results
                    df = pd.DataFrame(result["predictions"])
                    df["confidence"] = (df["confidence"] * 100).round(2).astype(str) + "%"
                    st.success("✅ Batch Prediction Results")
                    st.dataframe(df)
                else:  # Single result
                    label = result["predicted_label"].capitalize()
                    confidence = round(result["confidence"] * 100, 2)
                    st.success(f"**Predicted Label:** {label}")
                    st.progress(int(confidence))
                    st.write(f"**Confidence:** {confidence}%")

            else:
                st.error("❌ Error: Could not get prediction from API.")

        except Exception as e:
            st.error(f"⚠️ Error connecting to API: {e}")
