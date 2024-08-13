import streamlit as st
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=api_key)


llm_text = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=api_key,
    temperature=0.7,
    max_tokens=300,
    timeout=30,
    max_retries=2,
)


prompt_text = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a chatbot designed to provide comprehensive information about quitting drugs. For each drug, provide the following:\n"
            "1. Common withdrawal symptoms.\n"
            "2. Medications that can help manage withdrawal.\n"
            "3. Exercises and lifestyle changes to support recovery.\n"
            "4. Encourage consulting healthcare professionals for personalized advice.",
        ),
        ("human", "{drug}"),
    ]
)

# Function to process text input
def process_text_input(drug_name):
    chain = prompt_text | llm_text
    input_data = {"drug": drug_name}
    result = chain.invoke(input_data)
    return result.content


def get_image_description(image_data):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content(["describe the image as if you are a professional pharmacist", image_data])
    return response.text


def main():
    st.set_page_config(page_title="Multimodal Chatbot", page_icon="💬", layout="wide")

    st.title("💬 Multimodal Chatbot for Drug Information")
    st.markdown(
        """
        This application processes both text and image inputs to provide drug information 
        and analyze medical reports. Simply enter the drug name or upload an image for analysis.
        """
    )
    
    st.sidebar.title("Input Options")
    input_type = st.sidebar.selectbox("Select Input Type", ["Text", "Image"])

    
    if input_type == "Text":
        st.header("Text Input")
        drug_name = st.text_input("Enter the drug name:", placeholder="e.g., Nicotine, Alcohol")
        
        if st.button("Submit Text"):
            if drug_name:
                with st.spinner("Processing..."):
                    result = process_text_input(drug_name)
                st.success("Here's the information:")
                st.write(result)
            else:
                st.warning("Please enter a drug name.")

    elif input_type == "Image":
        st.header("Image Input")
        src_img = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
        
        if src_img is not None:
            img = Image.open(src_img)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            with st.spinner("Analyzing image..."):
                description = get_image_description(img)
            
            st.success("Image analysis completed!")
            st.write("**Description:**")
            st.write(description)
        else:
            st.info("Please upload an image file.")

if __name__ == "__main__":
    main()
