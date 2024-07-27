import streamlit as st
import requests
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


@tool
def get_available_courses(query: str) -> str:
    """Get available courses"""
    response = requests.get(
        "https://backend-produc.herokuapp.com/api/v1/cursos")
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching courses: {response.status_code} - {response.text}"


tools = [get_available_courses]

# Initialize the language model
llm = ChatOpenAI(model="gpt-4-0613", temperature=0)

# Initialize agent with tools and memory
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)
agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                         verbose=True,
                         memory=memory)

# Load and process documents
loader = TextLoader("./ragt.txt", encoding="UTF-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200,
                                               add_start_index=True)
all_splits = text_splitter.split_documents(docs)

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 6})

# Define system prompt
system_prompt = """Es a Ada a assistente virtual da Buka. Resume todas as suas respostas em poucas linas e mostre os cursos somente quando te perguntarem \
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

# Define the main chain
def rag_chain(query: str, chat_history: list):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    context = format_docs(docs)

    # Run the agent
    result = agent.run(
        input=f"{system_prompt.format(context=context)}\n\nHuman: {query}",
        chat_history=chat_history)

    return result


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Streamlit UI
st.set_page_config(page_title="Buka Cursos Chat")
st.header("Chat com Ada - Assistente da Buka")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        st.chat_message("assistant").write(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user").write(message.content)

# User input
user_query = st.chat_input("Escreva a sua mensagem aqui...")

# Only process if there's user input
if user_query:
    st.chat_message("user").write(user_query)
    
    with st.spinner("A processar..."):
        response = rag_chain(user_query, st.session_state.chat_history)
    
    st.chat_message("assistant").write(response)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--streamlit":
        import streamlit.cli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(streamlit.cli.main())
