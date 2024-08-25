import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message
import spacy

load_dotenv()
nlp = spacy.load('en_core_web_md')


def spacy_embedding(text):
    doc = nlp(text)
    return doc.vector


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path) 
    if not os.path.exists(directory) and directory:
        os.makedirs(directory)


def extract_text_from_files(files):
    combined_text = ""
    for file in files:
        ext = os.path.splitext(file.name)[1]
        if ext == ".pdf":
            combined_text += extract_text_from_pdf(file)
        elif ext == ".docx":
            combined_text += extract_text_from_docx(file)
        else:
            st.error("Unsupported file type")
    return combined_text


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text


def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=900, chunk_overlap=100, length_function=len
    )
    return text_splitter.split_text(text)


def respond_to_query(query):
    vector_store = st.session_state.vector_store
    if vector_store is None:
        st.error("Vector store is not available.")
        return

    query_embedding = spacy_embedding(query)
    chunk_embeddings = vector_store['embeddings']
    cosine_similarities = cosine_similarity(
        [query_embedding], chunk_embeddings).flatten()

    most_similar_chunk_index = cosine_similarities.argmax()
    most_similar_chunk = st.session_state.text_chunks[most_similar_chunk_index]

    st.session_state.chat_history.append(
        {"user": query, "bot": most_similar_chunk})
    display_chat_history()


def display_chat_history():
    with st.container():
        for i, messages in enumerate(st.session_state.chat_history):
            message(messages["user"], is_user=True, key=f"user_{i}")
            message(messages["bot"], key=f"bot_{i}")


def main():
    st.header("Chat with Your Documents")
    uploaded_files = st.file_uploader("Upload your file", type=[
                                      'pdf', 'docx'], accept_multiple_files=True)
    process_files = st.button("Process Files")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if process_files:
        combined_text = extract_text_from_files(uploaded_files)
        st.session_state.text_chunks = split_text_into_chunks(combined_text)
        st.write("Files processed and text chunks created.")

        store_name = "vector_store"
        base_path = os.path.abspath("saved_vector_stores")
        vector_store_file = os.path.join(base_path, f"{store_name}.pkl")
        ensure_directory_exists(vector_store_file)

        if os.path.exists(vector_store_file):
            try:
                with open(vector_store_file, "rb") as f:
                    st.session_state.vector_store = pickle.load(f)
                st.write('Vector store loaded from the disk.')
            except Exception as e:
                st.error(f"Error loading vector store: {e}")
                st.session_state.vector_store = None
        else:
            embeddings = [spacy_embedding(chunk)
                          for chunk in st.session_state.text_chunks]
            try:
                vector_store = {
                    'chunks': st.session_state.text_chunks, 'embeddings': embeddings}
                with open(vector_store_file, "wb") as f:
                    pickle.dump(vector_store, f)
                st.session_state.vector_store = vector_store
                st.write('Vector store saved to the disk.')
            except Exception as e:
                st.error(f"Error saving vector store: {e}")

    if st.session_state.text_chunks:
        user_query = st.text_input("Ask a question about your files:")
        if user_query:
            respond_to_query(user_query)


if __name__ == "__main__":
    main()
