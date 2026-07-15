import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Set your OpenAI API Key here
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

# Initialize database path
DB_DIR = "./my_personal_db"

st.title("💬 My Personal AI Chat")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- SIDEBAR: ADD/DELETE INFORMATION ---
with st.sidebar:
    st.header("Manage Knowledge")

    # ADD INFO
    st.subheader("Add Information")
    uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
    text_input = st.text_area("Or type info directly:")

    if st.button("Add to AI Memory"):
        embeddings = OpenAIEmbeddings()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)

        new_docs = []
        if uploaded_file:
            new_docs.extend(text_splitter.split_text(
                uploaded_file.getvalue().decode("utf-8")))
        if text_input:
            new_docs.append(text_input)

        if new_docs:
            # Add to ChromaDB
            db = Chroma.from_texts(new_docs, embeddings,
                                   persist_directory=DB_DIR)
            st.success(f"Added {len(new_docs)} chunks of info to memory!")

    # DELETE INFO
    st.subheader("Delete Information")
    if st.button("Clear All Memory"):
        # Deletes the entire database folder (simplistic approach)
        import shutil
        try:
            shutil.rmtree(DB_DIR)
            st.success("All memory wiped!")
        except FileNotFoundError:
            st.warning("No memory found to delete.")

# --- MAIN CHAT AREA ---
# Load the database
try:
    embeddings = OpenAIEmbeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = db.as_retriever()

    # Set up the AI Model
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    system_prompt = (
        "You are a helpful assistant. Use the following pieces of retrieved "
        "context to answer the question. If you don't know the answer, say that "
        "you don't know.\n\nContext: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the RAG chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about your personal info..."):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Get answer from AI
            response = rag_chain.invoke({"input": prompt})
            st.markdown(response["answer"])
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response["answer"]})

except Exception as e:
    st.info("Add some information in the sidebar to start chatting!")
