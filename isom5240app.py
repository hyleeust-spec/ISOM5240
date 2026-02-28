
# import part
from transformers import pipeline
from PIL import Image
import streamlit as st


# function part
def ageClassifier(imgFilename, modelName):
    # Load the age classification pipeline
    # The code below should be placed in the main part of the program
    age_classifier = pipeline("image-classification", 
                              model=modelName)
    
    image_name = imgFilename
    image_name = Image.open(image_name).convert("RGB")
    
    # Classify age
    age_predictions = age_classifier(image_name)

    return age_predictions


def output_msg(age_predictions):
    st.write(age_predictions)
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    
    # Display results
    st.write("Predicted Age Range:")
    st.write(f"Age range: {age_predictions[0]['label']}")
    
    st.write("Done")    



def main():
    # Streamlit UI
    st.header("Title: Age Classification using ViT")
    


#    age_predictions = ageClassifier("middleagedMan.jpg",
#                                   "prithivMLmods/Age-Classification-SigLIP2")
    age_predictions = ageClassifier("middleagedMan.jpg",
                                   "akashmaggon/vit-base-age-classification")
    output_msg(age_predictions)

  



# main part
if __name__ == "__main__":
    main()
