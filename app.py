# 패키지 임포트
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.callbacks import get_openai_callback
import os
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


# 텍스트 요약 함수
def summarize_document(document):
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    summary_prompt = "다음 텍스트에서 어떤 제품에 대한 설명서인지 간략히 요약해 주세요:\n\n" + document
    summary = llm.predict(summary_prompt)
    return summary


# 데이터 업로드 및 크로마DB 저장
def init(uploaded_file):
    if uploaded_file is not None:
        # 파일 저장 후 로드
        file_path = f"./temp/{uploaded_file.name}"
        os.makedirs("./temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 데이터 로드 (PDF 파일)
        loader = PyPDFLoader(file_path)
        document = loader.load()
        
        # 데이터 분할
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = text_splitter.split_documents(document)
        db = getDB()
        
        # Chroma DB에 저장
        db.add_documents(texts)
        st.success("PDF 파일이 성공적으로 처리되었습니다!")

        # 텍스트 요약
        chunk = '\n'.join([text.page_content for text in texts])
        summary = summarize_document(chunk[:3000])
        
        # 사이드바에 요약 표시
        st.sidebar.subheader("📜 PDF 요약")
        st.sidebar.write(summary)


# 텍스트 임베딩
def getDB():
    # 저장 및 검색
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(
        persist_directory="./db",  # 데이터베이스 경로
        embedding_function=embeddings
    )
    return docsearch


# RAG 체인 생성
def getRagChain():
    retriever = getDB().as_retriever()
    
    # rag_prompt = hub.pull("rlm/rag-prompt")
    rag_prompt = RunnableLambda(lambda x: f"""
    당신은 사용자 매뉴얼을 안내하는 AI 어시스턴트입니다. 사용자의 질문에 대해 명확하고 자세한 답변을 제공하세요.
    
    ### [컨텍스트]
    {x['context']}
    ### [질문]
    {x['question']}
    
    - 질문에 대해 완전한 문장으로 답변, 단답형 답변은 지양하고, 문장으로 명확하게 설명할 것.
    - 이상하거나 무의미한 질문 또는 매뉴얼과 없는 질문에는 단호하게 답변하지 말 것 예시 : 핸드폰 파손 방법, 핸드폰으로 라면 끓이기 
    - 아이콘(icon)에 대한 설명이 포함된 경우, 아이콘의 모양과 특징을 구체적으로 서술할 것.
    - 사용자가 명확한 답변을 얻을 수 있도록 조리 있게 정리하여 답할 것.
    """)

    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt 
        | llm 
    )
    return rag_chain


# 질문에 대한 답변 생성
def generate_answer(question):
    rag_chain = getRagChain()
    answer = rag_chain.invoke(question).content
    return answer



# 페이지 세팅
st.set_page_config(page_title="SKNETWORKS-FAMILY-AICAMP/SKN07-3rd-2Team", layout='wide')
st.title('📱 스마트폰 사용메뉴얼 기반 Q&A')
# st.header('제품: Samsung S25')

# 파일 업로드
with st.sidebar:
    uploaded_file = st.file_uploader("🗂️ PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded_file:
        init(uploaded_file)

# 초기화
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
    
with st.expander("질문&답변 히스토리 보기", expanded=False):
    for q, a in st.session_state.conversation:
        with st.chat_message('user'):
            st.write(q)
        with st.chat_message('assistant'):
            st.write(a)

# 프롬프트 입력 box
question = st.chat_input('질문을 입력하세요')
if question:
    with st.chat_message('user'):
        st.write(question)
    
    with st.spinner('답변을 생성 중입니다...'):
        with get_openai_callback() as cost:
            answer = generate_answer(question)
            with st.chat_message('assistant'):
                st.write(answer)
            
            st.session_state.conversation.append((question, answer))
