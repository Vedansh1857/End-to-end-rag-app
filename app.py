import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.cassandra import Cassandra
import cassio


from dotenv import load_dotenv
load_dotenv()
groq_api_key=os.environ['GROQ_API_KEY']


## connection of the ASTRA DB
ASTRA_DB_APPLICATION_TOKEN="AstraCS:OdhZdxwxZIHbikiHdoYMWDMu:80f5b5a6cb99da4ae1077bd4513098c975c760ea6c1caf12b85e051a3aa30d38" # enter the "AstraCS:..." string found in in your Token JSON file"
ASTRA_DB_ID="1a3d3f4b-d6e5-4fd5-87f7-65b2d76eaa02"
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID)


from langchain_community.document_loaders import WebBaseLoader
import bs4
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header")

                     )))
text_documents=loader.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
docs=text_splitter.split_documents(text_documents)


## Convert Data Into Vectors and store in AstraDB
os.environ["HUGGINGFACE_ACCESS_TOKEN"]=os.getenv("HUGGINGFACE_ACCESS_TOKEN")
## Embedding Using Huggingface
embeddings=HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
)
astra_vector_store=Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)


from langchain.indexes.vectorstore import VectorStoreIndexWrapper
astra_vector_store.add_documents(docs)
print("Inserted %i headlines." % len(docs))
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)


llm=ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")


print(astra_vector_index.query("Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique",llm=llm))
print("Querying from the vector")


from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
retriever=astra_vector_store.as_retriever()
document_chain=create_stuff_documents_chain(llm,prompt)
retrieval_chain=create_retrieval_chain(retriever,document_chain)


prompt = st.text_input("Enter your query here.")

if prompt:
    response=retrieval_chain.invoke({"input":prompt})
    st.write(response)
    print(f"Response of the retriever : \n {response}")
    print(f"Answer is : \n {response['answer']}")
