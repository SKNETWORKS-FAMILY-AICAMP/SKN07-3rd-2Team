# 패키지 임포트
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
import os

from DBClient import DBClient
from GptAgent import GptAgent

# Croma DB 접속용 클라이언트 인스턴스화
db_client = DBClient()
gpt_agent = GptAgent(retriever=db_client.get_retriever())

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
        db = db_client.get()
        
        # Chroma DB에 저장
        db.add_documents(texts)
        st.success("PDF 파일이 성공적으로 처리되었습니다!")

        # 텍스트 요약
        chunk = '\n'.join([text.page_content for text in texts])
        summary = summarize_document(chunk[:3000])
        
        # 사이드바에 요약 표시
        st.sidebar.subheader("📜 PDF 요약")
        st.sidebar.write(summary)

# 초기화
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# 페이지 세팅
st.set_page_config(page_title="SKNETWORKS-FAMILY-AICAMP/SKN07-3rd-2Team", layout='wide')
st.title('📱 스마트폰 사용메뉴얼 기반 Q&A')
# st.header('제품: Samsung S25')

# 파일 업로드
with st.sidebar:
    uploaded_file = st.file_uploader("🗂️ PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded_file:
        init(uploaded_file)

with st.container():
    with st.expander("질문&답변 히스토리 보기", expanded=False):
        for q, a in st.session_state.conversation:
            with st.chat_message('user'):
                st.write(q)
            with st.chat_message('assistant'):
                st.write(a)

with st.container():
    # 프롬프트 입력 box
    question = st.chat_input('질문을 입력하세요')
    if question:
        with st.chat_message('user'):
            st.write(question)
        
        with st.chat_message('assistant'):
            with st.spinner('답변을 생성 중입니다...'):
                answer = gpt_agent.send_message(question)
                st.session_state.conversation.append((question, answer))
                st.write(answer)
            
