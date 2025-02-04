from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from DBClient import DBClient
import os

class PdfUploader():
    __ROOT_PATH = './db/temp'
    def __init__(self,
                 model_name:str = 'gpt-4o-mini-2024-07-18'
                 ):
        os.makedirs(PdfUploader.__ROOT_PATH, exist_ok=True)
        self.__llm = ChatOpenAI(
            model_name=model_name, 
            temperature=0
            )
        self.__splitter = CharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=50
        )

        self.__db_client = DBClient()
        pass

    def upload(self, __file):
        file_path = self.__save_file(__file)
        __document = None
        __summary = None
        try:
            __document = self.__add_to_db(file_path)
            __summary = self.__summarize_document(__document)
        except Exception as e:
            print(f'Error:\n\t{e}')
        finally:
            self.__delete_file(file_path)
        
        return True if __document is not None else False , __summary
    
    # 텍스트 요약 함수
    def __summarize_document(self, documents):
        if documents is None:
            return None
        
        # 텍스트 요약
        chunk = '\n'.join([document.page_content for document in documents])
        
        summary_prompt = "다음 텍스트에서 어떤 제품에 대한 설명서인지 간략히 요약해 주세요:\n\n" + chunk[:3000]
        summary = self.__llm.predict(summary_prompt)
        return summary

    # Upload 파일 삭제
    def __delete_file(self, file_path:str=None):
        if file_path is None:
            return
        
        if os.path.exists(file_path):
            os.remove(file_path)

    # Upload 파일 저장
    def __save_file(self, __file):
        if __file is None:
            return None
        
        file_path = f"{PdfUploader.__ROOT_PATH}/{__file.name}"

        with open(file_path,'wb') as f:
            f.write(__file.getbuffer())

        return file_path
    
    # DB에 저장
    def __add_to_db(self, file_path:str=None):
        if file_path is None:
            return None
        
        loader = PyPDFLoader(file_path)
        document = loader.load()
        documents = self.__splitter.split_documents(document)
        self.__db_client.add(documents)
        
        return documents
