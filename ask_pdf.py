from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import uuid
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

load_dotenv()

def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64

def get_tables(chunks):
    tables = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Table" in str(type(el)):
                    tables.append(el.metadata.text_as_html)
    return tables

def summarize_non_image(elements):

    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOllama(temperature=0.5, model="llama3.1:8b")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    text_summaries = summarize_chain.batch(elements, {"max_concurrency": 3})
    return text_summaries

def summarize_image(images):
    prompt_template = """
    Describe the image in detail. For context, the image is part of a research paper explaining the 
    transformers architecture. Be specific about graphs, such as bar plots.
    """

    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()
    
    # Format images for batch processing
    formatted_images = [{"image": img} for img in images]
    image_summaries = chain.batch(formatted_images)
    return image_summaries

def retriever_init():
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return retriever

def add_docs(retriever, contents, summaries):
    id_key = "doc_id"
    
    doc_ids = [str(uuid.uuid4()) for _ in contents]
    summary_docs = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) 
        for i, summary in enumerate(summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, contents)))

    return retriever

def generate_answer(question, docs):
    """Generate a well-crafted answer based on retrieved documents."""
    
    # Format the retrieved documents
    context = ""
    for i, doc in enumerate(docs, 1):
        if hasattr(doc, 'page_content'):
            context += f"Document {i}:\n{doc.page_content}\n\n"
        else:
            context += f"Document {i}:\n{str(doc)}\n\n"
    
    prompt_template = """
    You are an AI assistant that provides accurate and helpful answers based on the given context.
    
    Context from PDF documents:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer the question based ONLY on the provided context
    - If the context doesn't contain enough information to answer the question, say so clearly
    - Be concise but comprehensive
    - Use specific details from the context when available
    - If there are tables, images, or charts mentioned in the context, reference them appropriately
    
    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    chain = prompt | model | StrOutputParser()
    
    answer = chain.invoke({"context": context, "question": question})
    return answer

