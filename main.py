import streamlit as st
import PyPDF2
import textwrap
import openai

import google.generativeai as genai
from io import BytesIO
import os
from PIL import Image as PILImage

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure the Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


client = openai.OpenAI(api_key="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")
# Function to summarize PDF using OpenAI (with chunking)
def summarize_text(pdf_text, max_tokens=500):
    
    summary = ""
    text_chunks = textwrap.wrap(pdf_text, 5000)  # Split text into chunks of 5000 characters

    for chunk in text_chunks:
        prompt = f"Summarize this document chunk in a concise and informative way (maximum {max_tokens} tokens):\n{chunk}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300)
        summary += response.choices[0].message.content.strip() + "\n"  # Append each chunk's summary

    return summary

def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

def summarize_image(image):
    return get_gemini_response("", image)
def main():
    st.title("Cabgen Summarization App")

    # First dropdown to select type of data to summarize
    summarization_type = st.selectbox("Select type of data to summarize:",
                                      options=['Text', 'Image', 'PDF'])

    # Second dropdown dynamically changes based on first dropdown selection
    if summarization_type == 'Text':
        text_to_summarize = st.text_area("Enter text to summarize:")
        if st.button("Summarize"):
            summary = summarize_text(text_to_summarize)
            st.subheader("Summary:")
            st.text_area(label="", value=summary, height=300)

    elif summarization_type == 'Image':
        image_type = st.selectbox("Select type of image upload:",
                                  options=['Single Image', 'Multiple Images'])

        if image_type == 'Single Image':
            uploaded_image = st.file_uploader("Upload a single image file", type=['jpg', 'jpeg', 'png'])
            if uploaded_image is not None:
                image = PILImage.open(uploaded_image)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                if st.button("Summarize"):
                    summary = summarize_image(image)
                    st.subheader("Image Caption:")
                    st.write(summary)

        elif image_type == 'Multiple Images':
            uploaded_files = st.file_uploader("Upload multiple image files", accept_multiple_files=True,
                                              type=['jpg', 'jpeg', 'png'])
            if uploaded_files:
                image_file_names = [file.name for file in uploaded_files]
                selected_image = st.selectbox("Select image to summarize:", options=image_file_names)

                if st.button("Summarize"):
                    selected_file = next(file for file in uploaded_files if file.name == selected_image)
                    image = PILImage.open(selected_file)
                    st.image(image, caption=selected_image, use_column_width=True)
                    summary = summarize_image(image)  # Placeholder for actual image summarization
                    st.subheader(f"Summary of {selected_image}:")
                    st.write(summary)
    elif summarization_type == 'PDF':
        pdf_type = st.selectbox("Select type of PDF upload:",
                                options=['Single PDF', 'Multiple PDFs'])

        if pdf_type == 'Single PDF':
            uploaded_file = st.file_uploader("Upload a single PDF file", type=['pdf'])
            if uploaded_file is not None:
                pdf_text = extract_pdf_text(uploaded_file)
                st.write(f"Summarizing {uploaded_file.name}...")
                if st.button("Summarize"):
                    summary = summarize_text(pdf_text)
                    st.subheader(f"Summary of {uploaded_file.name}:")
                    st.text_area(label="", value=summary, height=300)

        elif pdf_type == 'Multiple PDFs':
            uploaded_files = st.file_uploader("Upload multiple PDF files", accept_multiple_files=True,
                                              type=['pdf'])
            if uploaded_files:
                pdf_file_names = [file.name for file in uploaded_files]
                selected_pdf = st.selectbox("Select PDF to summarize:", options=pdf_file_names)

                if st.button("Summarize"):
                    selected_file = next(file for file in uploaded_files if file.name == selected_pdf)
                    pdf_text = extract_pdf_text(selected_file)
                    summary = summarize_text(pdf_text)
                    st.subheader(f"Summary of {selected_pdf}:")
                    st.text_area(label="", value=summary, height=300)

if __name__ == "__main__":
    main()