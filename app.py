from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from IPython.display import Markdown
import textwrap

# Load environment variables
load_dotenv()  # Load environment variables from .env file

# Configure generative AI with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Helper function to format response as Markdown
def to_markdown(text):
    text = text.replace('•', '  *')  # Example formatting, adjust as needed
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Function to get response from GenerativeModel
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")

background = '''
<style>
body {
    background-image: url('https://previews.123rf.com/images/sdecoret/sdecoret1709/sdecoret170901746/86924478-businessman-on-blurred-background-chatting-with-chatbot-application-3d-rendering.jpg');
    background-size: cover;
    background-repeat: no-repeat;
}
</style>
'''
st.markdown(background, unsafe_allow_html=True)
st.header("Chat With Rajesh")

# Input field for the question
input = st.text_input("Ask the question: ")


# Button to submit the question
submit = st.button("Get Answer")

# Check if clear history button is clicked
if st.button("Clear Input History"):
    st.session_state.input_history = []


if submit and input:
    response = get_gemini_response(input)
    st.subheader("Response:")
    st.markdown(response)


