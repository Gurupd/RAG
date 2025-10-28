import streamlit as st
st.set_page_config(page_title="Streamlit demo")
st.title("Demo app")
st.write('Welcome to first StreamLit app')
file_upload = st.file_uploader(
    "Choose a file",
    type=["txt","pdf"],
    accept_multiple_files=True
    )

# The st.write() function in Streamlit is used to display text or other output on the app interface
# if file_upload:
#     for upload_file in file_upload:
#         st.write(f"{upload_file.name} uploaded sucessfully" )

query_text=st.text_input("Enter your question",value="Enter your question here")
st.write(f"the question is : {query_text}")

# The st.form function lets you create a form container that groups multiple input widgets together. 

with st.form(key="qa_form",clear_on_submit=True,border=False):
    submitted=st.form_submit_button("submit")
    if submitted:
       st.write("Please upload a file and enter your question.")