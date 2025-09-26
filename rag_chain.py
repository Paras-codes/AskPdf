from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()

from config import PERSIST_DIR

qa_chain = None

# Prompt Template
PROMPT_TEMPLATE = """You are a helpful assistant. 
Use the following context to answer the question.
If the answer is not in the context, say 'I don’t know'.

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def get_vectordb():
    embeddings = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def get_rag_chain():
    global qa_chain
    if qa_chain is None:
        vectordb = get_vectordb()
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k":3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    return qa_chain
