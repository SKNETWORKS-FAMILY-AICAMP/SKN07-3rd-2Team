from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

class GptAgent():
    __TEMPLATE = """
    당신은 사용자 매뉴얼을 안내하는 AI 어시스턴트입니다. 사용자의 질문에 대해 컨텍스트를 바탕으로 명확하고 자세한 답변을 제공하세요.

    ### [컨텍스트]
    {context}

    ### [질문]
    {question}

    - 질문에 대해 완전한 문장으로 답변, 단답형 답변은 지양하고, 문장으로 명확하게 설명할 것.
    - 이상하거나 무의미한 질문에는 답변하지 마세요.
    - 컨텍스트에 없는 질문에는 단호하게 답변하지 말 것 예시 : 핸드폰 파손 방법, 핸드폰으로 라면 끓이기 
    - 아이콘(icon)에 대한 설명이 포함된 경우, 아이콘의 모양과 특징을 구체적으로 서술할 것.
    - 사용자가 명확한 답변을 얻을 수 있도록 조리 있게 정리하여 답할 것.
    """
    def __init__(self,
                 chat_model:str="gpt-4o-mini-2024-07-18",
                 retriever
                 ):
        self.__llm:ChatOpenAI = ChatOpenAI(
            model = chat_model,
            temperature=0
        )
        self.__prompt:PromptTemplate = PromptTemplate(
            template=GptAgent.__TEMPLATE,
            input_variables=['context','question']
        )
        self.__retriever = retriever
        self.__chain()

    def get_model_name(self):
        return self.__model_name
        
    def llm(self, llm=None):
        if llm != None:
            self.__llm = llm

        self.__chain()
        return self.__llm
    
    def retriever(self, retriever=None):
        if retriever != None:
            self.__retriever = retriever
        self.__chain()  
        return retriever
    
    def __chain(self, chain=None):
        if chain != None:
            self.__chain = chain
            return self.__chain

        self.__chain = RetrievalQA.from_chain_type(
            llm=self.__llm, 
            chain_type="stuff", 
            retriever=self.__retriever, 
            chain_type_kwargs={"prompt": self.__prompt}
        )
        return self.__chain

    def send_message(self,
                     question:str)-> str:
        
        __transaction = self.__chain.invoke(question)
        return __transaction['result']
