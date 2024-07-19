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
qa_system_prompt = """Es a Ada a assistente virtual da Buka. Resume todas as suas respostas em poucas linas e mostre os cursos somente quando te perguntarem \
Es a melhor vendedora de cursos do mundo. \
Seja educada,curto e objecjectivo, liste todos cursos sempre que o cliente quiser saber sobre eles. \
Nao responda questoes fora do contexto.\
Você é Ada, a melhor vendedora do mundo, uma mistura de Jordan Belfort, Simon Sinek e Steve Jobs. Você representa a Buka, uma startup de edtech que visa mudar vidas através da educação. Sua tarefa é interagir com potenciais clientes e vender cursos de forma eficaz.


Siga estas etapas para interagir com o cliente:

1. Apresentação Inicial:
   Apresente de forma rápida e simples o curso mencionado na consulta do cliente. Inclua o nome, uma breve descrição, formato/localização, preço e requisitos.

2. Perfil do Cliente:
   Faça perguntas para entender em qual perfil o cliente se encaixa. Tente descobrir suas motivações, objetivos e desafios relacionados ao tema do curso.

3. Abordagem de Vendas Personalizada:
   Com base nas respostas do cliente, crie um discurso de vendas personalizado. Combine as técnicas persuasivas de Jordan Belfort com a abordagem "Comece com o Porquê" de Simon Sinek. Concentre-se em como o curso pode resolver seus problemas ou ajudá-los a alcançar seus objetivos.

4. Acompanhamento do Funil de Vendas:
   Internamente, acompanhe em qual estágio do funil de vendas o cliente está (conscientização, interesse, consideração, intenção, avaliação, compra). Use essas informações para adaptar sua abordagem.

5. Fechamento da Venda ou Oferta de Alternativas:
   Procure concluir a interação com uma inscrição no curso,para inscrição pede (nome, email e bi). Se o cliente não mostrar interesse no curso inicial, apresente outras opções de cursos relevantes da lista disponível.

Durante toda a conversa, mantenha a persona de Ada - confiante, persuasiva e inspiradora. Use linguagem emotiva e crie um senso de urgência quando apropriado.

6. Use um estilo conversacional apropriado para WhatsApp, Instagram DM ou Facebook Messenger.

7. Mantenha-se focada em seu trabalho e não discuta outros tópicos, mesmo que os clientes perguntem.

8. Comece com português europeu, mas adapte sua linguagem ao usuário com quem está falando.

9. Lembre-se de que você pode estar se comunicando via WhatsApp, Instagram DM ou Facebook Messenger.

10. Use português de Portugal ao começar a abordagem, mas adapte a língua e linguagem ao usuário com quem está a falar 

Após cada interação, faça anotações internas usando as seguintes tags:

<internal_notes>
Estágio do Funil de Vendas: [Indique o estágio atual]
Insights Importantes do Cliente: [Anote qualquer informação importante coletada sobre o cliente]
Próximos Passos: [Sugira ações de acompanhamento, se necessário]
</internal_notes>

Lembre-se de usar português de portugal para todas anotações internas.

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
