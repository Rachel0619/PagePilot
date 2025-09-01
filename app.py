from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain.callbacks import get_openai_callback
import ask_pdf
from unstructured.partition.pdf import partition_pdf

def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your PDF')
    st.header('Ask your PDF 💬')

    # upload file
    pdf = st.file_uploader("Upload your PDF here", type="PDF")

    # extract text
    if pdf:

        chunks = partition_pdf(
            file=pdf,
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables

            extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

            chunking_strategy="by_title",          # or 'basic'
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,
        )

        if not chunks:
            st.error("Failed to extract content from PDF. Please try a different file.")
            return
            
        texts = []
        for chunk in chunks:
            texts.append(chunk)
        tables = ask_pdf.get_tables(chunks)
        images = ask_pdf.get_images_base64(chunks)

        # Only process non-empty lists
        text_summaries = ask_pdf.summarize_non_image(texts) if texts else []
        table_summaries = ask_pdf.summarize_non_image(tables) if tables else []
        image_summaries = ask_pdf.summarize_image(images) if images else []

        retriever = ask_pdf.retriever_init()
        
        # Only add documents if they exist
        if texts and text_summaries:
            retriever = ask_pdf.add_docs(retriever, texts, text_summaries)
        if tables and table_summaries:
            retriever = ask_pdf.add_docs(retriever, tables, table_summaries)
        if images and image_summaries:
            retriever = ask_pdf.add_docs(retriever, images, image_summaries)
            
        # Check if we have any documents
        total_docs = len(texts) + len(tables) + len(images)
        if total_docs == 0:
            st.error("No content could be extracted from the PDF. Please try a different file.")
            return

        user_question = st.text_input("Ask a question about your PDF")
        if user_question:
            # Retrieve relevant documents
            docs = retriever.invoke(user_question)
            
            # Generate answer using retrieved documents
            answer = ask_pdf.generate_answer(user_question, docs)
            st.write(answer)

if __name__=='__main__':
    main()

