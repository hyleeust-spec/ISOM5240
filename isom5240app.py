import streamlit as st
from transformers import pipeline
from PIL import Image

def age_classifier(image_file_name, models_name):
  # Classify age
  # Load the age classification pipeline
  # The code below should be placed in the main part of the program
  
  age_classifier = pipeline("image-classification",
                              model=models_name)
  
  image_name = image_file_name
  image_name = Image.open(image_name).convert("RGB")


def main():
  # Streamlit UI
  st.header("Title: Age Classification using ViT")
  
  age_classifier("middleagedMan.jpg", "prithivMLmods/Age-Classification-SigLIP2")
  
  age_predictions = age_classifier(image_name)
  st.write(age_predictions)
  age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
  
  # Display results
  st.write("Predicted Age Range:")
  st.write(f"Age range: {age_predictions[0]['label']}")
  
  st.write("Done")

main()
