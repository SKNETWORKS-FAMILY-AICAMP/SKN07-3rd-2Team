# SKN07-3rd-2Team
## LLM을 연동한 내외부 문서 기반 질의응답 시스템

---

# 🏃🏃‍♂️🏃‍♀️ 팀명 : 김성근과 아이들
|<img src="https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/%E3%85%85%E3%84%B1.jpg" alt="김성근" width="120"/>|<img src="https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/%E3%85%85%E3%85%81.jpg" alt="윤수민" width="120"/>|<img src="https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/%E3%85%88%E3%85%8A.jpg" alt="이재철" width="120"/>|
|---|---|---|
| <div align="center">**김성근**</div> | <div align="center">**윤수민**</div> | <div align="center">**이재철**</div> |
| <div align="center">Refactoring</div> | <div align="center">Streamlit</div> | <div align="center">RAG<br>Prompt Engineering</div> |

 ---
 
# 📜 제품 사용 설명서 질의응답 챗봇
## 🔖 프로젝트 소개
이 프로젝트는 LLM(Large Language Model)과 RAG(Retrieval-Augmented Generation) 방식을 활용하여 사용 설명서를 기반으로 한 질의응답(Q&A) 챗봇을 구축하는 것을 목표로 합니다.
## 🔖 프로젝트 동기
- 복잡한 사용 설명서를 읽기 어려워하는 사용자들을 위해 개발
- 필요한 정보를 빠르게 찾을 수 있도록 자동화된 Q&A 시스템 제공
- 고객 지원 부담 감소 및 사용자 경험 개선
- 다양한 제품에 적용 가능한 범용적인 솔루션 구축
## 🔖 프로젝트 목표
- LLM과 RAG 방식을 활용한 질의응답 챗봇 구축
- 사용 설명서에서 관련 정보 검색 및 자연어 응답 생성
- 누구나 쉽게 사용할 수 있는 직관적인 인터페이스 제공
- 다양한 제품 설명서 지원 (가전, 소프트웨어, 자동차 등)
- 지속적인 피드백을 통한 성능 개선

---

## 🔨 기술 스택
<div>
<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
<img src="https://a11ybadges.com/badge?logo=openai" alt="OpenAI" width="163" height="28"/>
<img src="https://img.shields.io/badge/langchain-F7DF1E?style=for-the-badge&logo=langchain&logoColor=black">
<img src="https://img.shields.io/badge/streamlit%20-%23FF0000.svg?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/chromadb.jpg" alt="chromadb" width="90" height="40">
<div>
</div>
<img src="https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white">
<img src="https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white">
<img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
<img src="https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white">
</div>
 
## 📌 코드 리뷰
### 데이터 로드 및 전처리 (PDF 처리 및 ChromaDB 저장)
```python
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

# 텍스트 임베딩
def getDB():
    # 저장 및 검색
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma(
        persist_directory="./db",  # 데이터베이스 경로
        embedding_function=embeddings
    )
    return docsearch
```
| ![pdf_file](https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/pdf_10page.jpg?raw=true) | ![data_load](https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/data_load.jpg?raw=true) |
|:-------------------------------------:|:-------------------------------------:|

### PDF 요약
```python
# 텍스트 요약 함수
def summarize_document(document):
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
    summary_prompt = "다음 텍스트에서 어떤 제품에 대한 설명서인지 간략히 요약해 주세요:\n\n" + document
    summary = llm.predict(summary_prompt)
    return summary
```

### RAG Chain 생성
```python
def getRagChain():
    retriever = getDB().as_retriever()
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
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | rag_prompt 
        | llm 
    )
    return rag_chain

def generate_answer(question):
    rag_chain = getRagChain()
    answer = rag_chain.invoke(question).content
    return answer
```

 ---
 
## 📌 수행결과(테스트/시연 페이지)
### 구현 화면
![screen](https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/%EC%A0%84%EC%B2%B4%ED%99%94%EB%A9%B4.jpg?raw=true)

---
![history](https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/%EC%A7%88%EB%AC%B8_%EB%8B%B5%EB%B3%80_history.jpg?raw=true)

---
### 이상 질문 시 답변
![weired](https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/%EC%9D%B4%EC%83%81%EC%A7%88%EB%AC%B8.jpg?raw=true)


---
## 📖 한 줄 회고
김성근 : 항상 시간은 짧은 것 같다

윤수민 : 아직 많이 부족하지만, 프로젝트를 통해 많이 배울 수 있었습니다. LLM 어렵지만 재밌지만 가까워지긴 힘든 자식,,

이재철 : 팀원 중 제일 부족함이 많은 프로젝트였습니다. 부족한 팀원 도움주시면서 하시느라 고생 많으셨습니다🙇‍♂️
