from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama  # or use HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import Qdrant
import fitz

client = OpenAI(
    api_key="AIzaSyBGiws8Lj8a_bCADDDH_W_SwmyVhkpnnjU",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

uploaded_file=st.file_uploader("Choose files",type=['txt','pdf','img'])
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

def read_pdf(file):
    pdf_document=fitz.open(stream=file.read(),filetype="pdf")
    text=""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        print(page.get_text())
        text += page.get_text()
    return text
def rag_response(file,query):
    if file is not None:
        if file.type=="application/pdf":
            documents=read_pdf(file)
            # return
        else:
            documents = file.read().decode('utf-8')
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts=text_splitter.create_documents([documents])
        embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # vector_store=Chroma.from_texts([doc.page_content for doc in texts],embedding=embeddings)
        # retriever = vector_store.as_retriever(search_kwargs={"k": 2})
        vector_store=Qdrant.from_documents(texts,embeddings,location=":memory:",collection_name="my_documents")
        retriever=vector_store.as_retriever()
        res=retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in res])

        resp=res[0].page_content
        custom_rag_prompt = ChatPromptTemplate.from_template("""
        You are an assistant answering based on the given context.    
          Context:
        {context}

        Question:
        {query}
        Answer in a concise and clear way.
        """).format(context=context,query=query)

        # st.text_area(res)
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant that answers based strictly on provided context."},
                {"role": "user", "content": custom_rag_prompt}
            ]
        )
        print(response.choices[0].message.content)

with st.form('myform', clear_on_submit=False, border=False):
    submitted = st.form_submit_button('Submit')
    if submitted and uploaded_file and query_text:
        rag_response(uploaded_file,query_text)
    
