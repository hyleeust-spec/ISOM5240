import streamlit as st
from transformers import pipeline

# Load the text classification model pipeline
classifier = pipeline("text-classification",model='isom5240ust/bert-base-uncased-emotion')

# Streamlit application title
st.title("Text Classification for you")
st.write("Classification for 6 emotions: sadness, joy, love, anger, fear, surprise")

# Text input for user to enter the text to classify
text = st.text_area("Enter the text to classify", "")

# Perform text classification when the user clicks the "Classify" button
if st.button("Classify"):
    # Perform text classification on the input text
    result = classifier(text)[0]

    # Display the classification result
    st.write("Text:", text)
    st.write("Label:", result['score'])
    st.write("Score:", result['label'])
