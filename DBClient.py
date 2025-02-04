from langchain.embeddings import OpenAIEmbeddings  #← OpenAIEmbeddings를 가져오기
from langchain.vectorstores import Chroma

class DBClient():
    def __init__(self, 
                 persist_directory:str = './db/Galaxy25',
                 embedding_model="text-embedding-ada-002"):
        self.__embeddings = OpenAIEmbeddings( #← OpenAIEmbeddings를 초기화
            model=embedding_model
        )
        self.__vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.__embeddings,
        )
        pass

    def get(self):
        return self.__vector_store
    
    def get_retriever(self):
        return self.__vector_store.as_retriever()

    def query(self, query):
        result = self.__vector_store.similarity_search(query=query)
        print(result.__len__())
        return "\n".join([x.page_content for x in result])

if '__main__' == __name__:
    db = DBClient()
    results = db.query('소셜')
    print(results)
