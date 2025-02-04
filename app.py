# 패키지 임포트
import streamlit as st

from DBClient import DBClient
from GptAgent import GptAgent
from pdf_util import PdfUploader

# Croma DB 접속용 클라이언트 인스턴스화
db_client = DBClient()
gpt_agent = GptAgent(retriever=db_client.get_retriever())
pdf_uploader = PdfUploader()

# 데이터 업로드 및 크로마DB 저장
def init(uploaded_file):
    if uploaded_file is None:
        return
    
    success, summary =  pdf_uploader.upload(uploaded_file)

    if success:
        st.success("PDF 파일이 성공적으로 처리되었습니다!")
    else:
        st.error("PDF 파일 처리에 실패했습니다!")
        return
    
    if summary is not None:
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
            
