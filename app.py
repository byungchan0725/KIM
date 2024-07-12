import streamlit as st
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

def model(file, text):
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    file_path = file 
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


UPLOAD_DIRECTORY = "./"

# 디렉토리가 없으면 생성
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Sidebar에서 파일 업로드
with st.sidebar:
    file = st.file_uploader("파일을 업로드해주세요.")

    if file:
        file_path = os.path.join(UPLOAD_DIRECTORY, file.name)
        
        # 파일을 지정된 경로에 저장
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        st.success(f"{file.name}이/가 성공적으로 업로드 되었습니다.", icon="✅")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("궁금한 점을 입력해주세요."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = model(file_path, prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})