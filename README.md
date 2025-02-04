# SKN07-3rd-2Team
## 3차 프로젝트: LLM을 연동한 내외부 문서 기반 질의응답 시스템

---

# 🏃🏃‍♂️🏃‍♀️ 팀명 : 토마스와 친구들
|김성근|윤수민|이재철|
|---|---|---|
|![김성근]()|![윤수민]()|![이재철]()|
|풀스택|풀스택|Langchain & RAG
streamlit|

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
## 🔨 기술 스택
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

![OpenAI]([https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white](https://camo.githubusercontent.com/b33fa4762f2df60c221d8790c9d89a49539157c8de82900a0dc0051c8ef7a92c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6f70656e61692d3037363941443f7374796c653d666f722d7468652d6261646765266c6f676f3d6f70656e6169266c6f676f436f6c6f723d626c61636b)), ![chromadb](https://private-user-images.githubusercontent.com/175914629/385558173-c8cd01e7-6ce6-46db-8cc3-b13286829cf3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mzg2MjczODIsIm5iZiI6MTczODYyNzA4MiwicGF0aCI6Ii8xNzU5MTQ2MjkvMzg1NTU4MTczLWM4Y2QwMWU3LTZjZTYtNDZkYi04Y2MzLWIxMzI4NjgyOWNmMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwMjAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDIwM1QyMzU4MDJaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05ZjRmOWQ5M2ZhNzQ5MGYxNjk2YzM2NmU4NzFkN2M5OWM5ZGRiYjIxYTQxOTM1Njk0ZjRiZGMwMTc0NDk4NzNjJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.hq1zjmmS2FwO65qyAfHcSfrSJdRFQkNi6tEayhQL4eI)![Langchain](https://camo.githubusercontent.com/4f7aaf07d9e13fd95b27d2db63e0712cfe0ed4588a6ac1b7b3cb505af6d37abe/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6c616e67636861696e2d4637444631453f7374796c653d666f722d7468652d6261646765266c6f676f3d6c616e67636861696e266c6f676f436f6c6f723d626c61636b)![streamlit](https://camo.githubusercontent.com/a79929766bd74e02c10f8a234c6037dacc4d0a1d5d73c4fc1bad339b253a82a7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f73747265616d6c69742532302d2532334646303030302e7376673f7374796c653d666f722d7468652d6261646765266c6f676f3d73747265616d6c6974266c6f676f436f6c6f723d7768697465)

![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)

---
## 📂시스템 아키텍처
![system]("https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/system.png")

## 🔖 주요 프로시저
데이털 파일 로



### 🔖 데이터 파일 로드, 분할, VectorDB 저장
```python
def init():
    # 데이터 로드 (PDF 파일)
    loader = PyPDFLoader("./your_pdf_file.pdf")
    document = loader.load()
    # 데이터 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(document)
    db = getDB()
    # Chroma DB 에 저장
    docsearch = db.add_documents(texts)
```
![data_load]("https://github.com/pladata-encore/SKN07-3rd-2Team/blob/main/image/data%20load.png")
 
## 🔖 수행결과(테스트/시연 페이지)
 
## 📖 한 줄 회고
김성근 : 

윤수민 : 

이재철 : 팀원 중 제일 부족함이 많은 프로젝트였습니다. 부족한 팀원 도움주시면서 하시느라 고생 많으셨습니다🙇‍♂️
