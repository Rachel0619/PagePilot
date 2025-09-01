from unstructured.partition.pdf import partition_pdf
import base64
from IPython.display import Image, display
from langchain_groq import ChatGroq
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
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
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

if __name__ == "__main__":
    output_path = "./content/"
    file_path = output_path + 'attention.pdf'

    chunks = partition_pdf(
        filename=file_path,
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

    texts = []
    for chunk in chunks:
        texts.append(chunk)
    tables = get_tables(chunks)
    images = get_images_base64(chunks)

    text_summaries = summarize_non_image(texts)
    table_summaries = summarize_non_image(tables)
    image_summaries = summarize_image(images)

    retriever = retriever_init()
    retriever = add_docs(retriever, texts, text_summaries)
    retriever = add_docs(retriever, tables, table_summaries)
    retriever = add_docs(retriever, images, image_summaries)

    # test
    docs = retriever.invoke(
        "who are the authors of the paper?"
    )
