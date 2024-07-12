# GPT는 PDF를 RAG하는 용도로 사용할 것. 
import os 

from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv() 


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

def GenerativeAnswerPdf(text):
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    file_path = PdfUploaded_GET() 
    loader = PyPDFLoader(file_path)
    docs = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    # 정보 검색 
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0,
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke(f"{text}")

    return response