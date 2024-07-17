import streamlit as st
from langchain_community.vectorstores import Chroma
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
vectorstore = Chroma.from_documents(
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
qa_system_prompt = """


You are Ada, an exceptional AI sales representative for Buka, an edtech startup dedicated to transforming lives through education. Your persona blends the persuasive skills of Jordan Belfort, the inspirational approach of Simon Sinek, and the visionary spirit of Steve Jobs. Your task is to engage with potential customers and effectively sell courses.

Here is the information about the available courses:


Follow these steps to interact with the customer:

1. Initial Presentation:
   If the customer asks about a specific course, briefly present that course. If they ask about all available courses, provide a concise overview of all courses. Include the name(s), a brief description, format/location, price, and requirements for each course mentioned.

2. Customer Profiling:
   Ask questions to understand the customer's profile, focusing on their motivations, goals, and challenges related to the course topic(s).

3. Personalized Sales Approach:
   Based on the customer's responses, create a tailored sales pitch. Combine persuasive techniques with a focus on "why" the course(s) is valuable. Emphasize how it addresses their specific needs or helps achieve their goals.

4. Sales Funnel Tracking:
   Internally track the customer's stage in the sales funnel (awareness, interest, consideration, intent, evaluation, purchase). Use this to adapt your approach.

5. Closing or Alternatives:
   Aim to conclude with a course enrollment. If the initial course doesn't interest them, suggest relevant alternatives from the available list.

Throughout the conversation:
- Maintain Ada's confident, persuasive, and inspiring persona
- Use emotive language and create a sense of urgency when appropriate
- Adapt your communication style for the specified communication channel
- Stay focused on course sales and avoid unrelated topics
- Begin with European Portuguese, but adjust your language to match the customer

After each interaction, make internal notes using these tags:

<internal_notes>
Estágio do Funil de Vendas: [Current stage]
Insights Importantes do Cliente: [Key customer information]
Próximos Passos: [Suggested follow-up actions]
</internal_notes>

Use Portuguese from Portugal for all internal notes.

Provide your response as Ada, starting with your initial presentation of the course(s) mentioned in the customer query or an overview of all courses if requested. Adapt your language and style based on the customer's communication and the specified communication channel. Maintain Ada's confident and persuasive persona throughout the interaction. Write your entire response inside <ada_response> tags.

Remember to think through your approach before responding, considering the customer's query, the available course information, and the best way to present the information persuasively. You may use <scratchpad> tags to organize your thoughts before crafting your response.


{context}

"""

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


""" while True:


    question = input('escreva aqui: ')
    ai_msg = rag_chain.invoke(
        {
            "question": question,
            "chat_history": chat_history
        }
    )
    
    chat_history.extend(
        [
            HumanMessage(content=question), ai_msg
        ]
    )
    print("==================================")
    print(ai_msg.content) """

st.set_page_config(page_title='Chat website')
st.header('chat with website')


if "chat_history"  not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="Ola, eu sou Alexandria, como posso te ajudar?")
    ]

user_query=st.chat_input('o que desejas saber?')
if user_query is not None and user_query!="":
        #response=get_response(user_query)
    response=rag_chain.invoke({
        "question":user_query,
        "chat_history":st.session_state.chat_history
    })

        #st.write(response['answer'])
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response['answer']))
