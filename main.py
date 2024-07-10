import streamlit as st
from langchain_community.vectorstores import Chroma,FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

#llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    #max_tokens=1024,
    #timeout=None,
    #max_retries=2,
    # other params...
)
### Construct retriever ###

loader = TextLoader('./cursos.txt',encoding='UTF-8')

docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True 
)
all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(
    documents = all_splits,
    embedding = embeddings
)
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={"k":6}
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

contextualize_q_chain.invoke(
    {
        "chat_history":[
            HumanMessage(content="What does LLM stand for?"),
            AIMessage(content="Large language model"),
        ],
        "question": "What is meant by large?",
    }
)
qa_system_prompt = """Es a Ada a assistente virtual da Buka. Resume todas as suas respostas em poucas linas e mostre os cursos somente quando te perguntarem \
Es a melhor vendedora de cursos do mundo. \
Seja educada,curto e objecjectivo. \
Nao responda questoes fora do contexto.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
def format_docs(docs):
    return "".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]
    
rag_chain = (
    RunnablePassthrough.assign(
        context = contextualized_question | retriever | format_docs
    )
    | qa_prompt 
    | llm
)

#chat_history = []



st.set_page_config(page_title='Buka chatbot')
st.header('BUKA assistente')

if "chat_history"  not in st.session_state:
    st.session_state.chat_history=[
    ]

user_query=st.chat_input('o que desejas saber?')
if user_query is not None and user_query!="":
        #response=get_response(user_query)
    response=rag_chain.invoke({
        "question":user_query,
        "chat_history":st.session_state.chat_history
    })

    st.session_state.chat_history.extend(
        [
            HumanMessage(content=user_query), response
        ]
    )
    #st.write(response)
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message('Ai'):
                st.write(message.content)
        if isinstance(message,HumanMessage):
            with st.chat_message('Human'):
                st.write(message.content)
