import streamlit as st
import re
import pickle
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

# Load model and tfidf
model = pickle.load(open('model.pkl', 'rb'))
tf = pickle.load(open('tfidf.pkl', 'rb'))

class Preprocessing:
    def __init__(self, data):
        self.data = data

    def text_preprocessing_user(self):
        lm = WordNetLemmatizer()
        review = re.sub('[^a-zA-Z0-9 ]', '', self.data)
        review = review.lower()
        review = review.split()
        review = [lm.lemmatize(x) for x in review if x not in stopwords]
        return [' '.join(review)]


# UI
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Title")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        processed = Preprocessing(user_input).text_preprocessing_user()
        vector = tf.transform(processed)
        prediction = model.predict(vector)

        if prediction[0] == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")