from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import Chroma
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title='Ask your PDF')
    st.header('Ask your PDF 💬')

    # upload file
    pdf = st.file_uploader("Upload your PDF here", type="PDF")

    # extract text
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_spliter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_spliter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = Chroma.from_texts(chunks, embeddings)

        user_question = st.text_input("Ask a question about your PDF")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = ChatOpenAI(model="gpt-4o-mini") 
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
            st.write(response)

if __name__=='__main__':
    main()

