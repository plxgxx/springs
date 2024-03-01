import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


def langchain_answer(file, question: str) -> str:
    load_dotenv()
    loader = PyPDFLoader(file_path = file)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300
    )
    documents = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index_pdf")

    new_vectorstore = FAISS.load_local("faiss_index_pdf", embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),  
        retriever=new_vectorstore.as_retriever()
    )

    response = qa.invoke(question)

    return response['result']