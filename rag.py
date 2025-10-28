from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # or use HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter

uploaded_file=st.file_uploader("Choose files",type=['txt','pdf','img'])
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

def rag_response(file,query):
    documents = [file.read().decode()]
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store=Chroma.from_texts(text_splitter,embedding=embeddings)
    print(vector_store)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    retriever.invoke(query)
    
with st.form('myform', clear_on_submit=False, border=False):
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    rag_response(uploaded_file,query_text)
    
