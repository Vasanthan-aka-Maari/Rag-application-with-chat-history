import streamlit as st
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os 
import requests

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="mixtral-8x7b-32768"
)

st.title("RAG Application with Chat History")
st.write("Enter the website url (1) to chat with")

session_id = st.text_input("Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

user_url = st.text_input("Enter 1 website url")

def check_website_status(url):
    response = requests.get(url, timeout=5)
    if response.status_code == 200:
        return True
    return False

if user_url:
    if check_website_status(user_url):
        loader = WebBaseLoader(user_url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(splits, embedding)
        retriever = vectorstore.as_retriever()

        contextualize_system_prompt = (
            """
            Given chat history and the latest user question which might refer context in the chat history, 
            formulate a standalone question which can be understood without chat history. Do not answer the question, 
            just reformulate the question.
            """
        )

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

        system_prompt = (
            """
            You are a helpful assistant answering questions tasks.
            Use the following piece of retrieved context to answer the question. If you don't konw the answer
            just say "I don't know." Keep the answer concise, preferrably in less than 3 sentences.

            {context}
            """
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # User Input
        user_input=st.text_input("User Input")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input}, 
                config={
                    "configurable": {"session_id":session_id}
                },
            )
            print(st.session_state.store)
            print("Assistant: ", response['answer'])
            print("Chat history: ", session_history.messages)
            for message in session_history.messages:
                if message.type == "human":
                    st.write("You: ", message.content)
                elif message.type == "ai":
                    st.write("Assistant: ", message.content)
    else:
        st.write("Invalid website url")
    



    


    
