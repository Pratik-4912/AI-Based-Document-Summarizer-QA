import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredEPubLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from sklearn.cluster import KMeans
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
import time

# ✅ Use your OpenAI API Key here (Make sure this file is not public)
openai_api_key =""

def load_book(file_obj, file_extension):
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_obj.read())
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file.name)
            pages = loader.load()
            text = "".join(page.page_content for page in pages)
        elif file_extension == ".epub":
            loader = UnstructuredEPubLoader(temp_file.name)
            data = loader.load()
            text = "\n".join(element.page_content for element in data)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
        os.remove(temp_file.name)
    return text.replace('\t', ' ')

def split_and_embed(text, batch_size=5):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=2000, chunk_overlap=500)
    docs = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectors, all_docs = [], []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        batch_vectors = embeddings.embed_documents([doc.page_content for doc in batch_docs])
        vectors.extend(batch_vectors)
        all_docs.extend(batch_docs)

    faiss_index = FAISS.from_documents(all_docs, embeddings)
    return docs, vectors, faiss_index

def cluster_embeddings(vectors, num_clusters):
    num_samples = len(vectors)
    if num_samples < num_clusters:
        num_clusters = num_samples
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    return sorted([np.argmin(np.linalg.norm(vectors - center, axis=1)) for center in kmeans.cluster_centers_])

def summarize_chunks(docs, selected_indices):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo-16k')
    prompt_template = PromptTemplate(
        template="""You are provided with a passage from a document. Your task is to produce a comprehensive summary of this passage. Ensure accuracy and avoid adding any interpretations or extra details not present in the original text. The summary should be at least three paragraphs long and fully capture the essence of the passage.
        ```{text}```
        SUMMARY:""",
        input_variables=["text"]
    )
    selected_docs = [docs[i] for i in selected_indices]
    summaries = []

    for doc in selected_docs:
        time.sleep(1)
        summary = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template).run([doc])
        summaries.append(summary)

    return "\n".join(summaries)

def create_final_summary(summaries):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=3000, model='gpt-4', request_timeout=120)
    prompt_template = PromptTemplate(
        template="""You are given a series of summarized sections from a document. Your task is to weave these summaries into a single, cohesive, and verbose summary. The reader should be able to understand the main events or points of the document from your summary. Ensure you retain the accuracy of the content and present it in a clear and engaging manner.
        ```{text}```
        COHESIVE SUMMARY:""",
        input_variables=["text"]
    )
    time.sleep(1)
    reduce_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
    return reduce_chain.run([Document(page_content=summaries)])

def create_qa_system(faiss_index):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=1000, model='gpt-3.5-turbo')
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=faiss_index.as_retriever())

def main():
    st.title("📚 Document Summarizer + Q&A")

    uploaded_file = st.file_uploader("Upload a PDF or EPUB file", type=["pdf", "epub"])

    if uploaded_file:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        if st.button("🔍 Generate Summary"):
            with st.spinner("⏳ Processing..."):
                text = load_book(uploaded_file, file_ext)
                docs, vectors, faiss_index = split_and_embed(text)
                selected_indices = cluster_embeddings(vectors, 5)
                summaries = summarize_chunks(docs, selected_indices)
                final_summary = create_final_summary(summaries)
                st.subheader("📝 Final Summary")
                st.write(final_summary)
                st.session_state.faiss_index = faiss_index

    if "faiss_index" in st.session_state:
        qa_chain = create_qa_system(st.session_state.faiss_index)
        question = st.text_input("❓ Ask a question about the document")
        if st.button("Ask"):
            with st.spinner("🔍 Searching..."):
                answer = qa_chain.run(question)
                st.subheader("💡 Answer")
                st.write(answer)

if __name__ == "__main__":
    main()

