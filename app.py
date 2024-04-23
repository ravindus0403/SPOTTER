import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Pre-download NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    nltk.download('punkt', download_dir=nltk_data_path)

ps = PorterStemmer()

def transform_text(Content):
    Content = Content.lower()  # Lowercase
    Content = nltk.word_tokenize(Content)  # Tokenization

    y = []
    for i in Content:
        if i.isalnum():
            y.append(i)  # Remove special characters

    Content = y[:]
    y.clear()

    for i in Content:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(ps.stem(i))  # Stemming

    return " ".join(y)

# Load the TfidfVectorizer from the saved file
try:
    with open('Vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
except FileNotFoundError:
    st.error("Vectorizer file not found. Please ensure it exists.")

# Load the model from the saved file
try:
    model = pickle.load(open('Model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Please ensure it exists.")

# Custom CSS for setting background image
custom_css = """
    body {
        background-image: url('https://app.gemoo.com/share/image-annotation/641000072215052288?codeId=MpO0Jm7qBNpBm&origin=imageurlgenerator&card=641000069337759744');
        background-size: cover;
    }
"""

# Set page configuration and inject custom CSS
st.set_page_config(
    page_title="SPOTTER",
    page_icon=":shield:",
    layout="wide"
)

# Inject custom CSS
st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

st.title("SPOTTER")

# Add subtitle
st.subheader("Digital Guardian Against SPAM Email/SMS")

input_sms = st.text_area("Enter the message", height=200)

if st.button('Predict') or (input_sms and input_sms.strip()):
    if not input_sms.strip():  # Check if input message is blank or contains only whitespace
        st.warning("Please add your message.")
    else:
        # 1. Preprocessing
        Transformed_Content = transform_text(input_sms)
        # 2. Vectorizing
        vector_input = tfidf.transform([Transformed_Content])
        # 3. Predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Prediction: Spam")
        else:
            st.header("Prediction: Not Spam")

        # Additional information
        st.subheader("Additional Information:")

        # Precision score as a percentage
        precision_score = model.predict_proba(vector_input)[0][1] * 100
        st.write(f"Precision Score: {precision_score:.2f}%")

