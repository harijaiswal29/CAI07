#%pip install streamlit
import streamlit as st

# Header
st.title(":blue[Chatbot for financial query of Google from their Statement]")

#file upload in different format
st.file_uploader("Upload a file", type=["csv", "txt","pdf"])
st.chat_input("Ask a question")

