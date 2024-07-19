import streamlit as st
from langchain_community.vectorstores import Chroma, FAISS, Cassandra
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# llm=ChatOpenAI(model="gpt-4o-mini-2024-07-18",temperature=0)

llm = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    #model="claude-3-haiku-20240307",
    temperature=0,
)

examples = [
    {
        "input": "Olá",
        "output": """"Olá! Bem-vindo à Buka, onde mudamos vidas por meio da educação. Sou a Ada, e estou aqui para ajudá-lo a encontrar o curso perfeito para impulsionar a sua carreira ou atingir seus objectivos. \n\nTemos vários cursos fantásticos disponíveis. Gostaria de saber mais sobre algum em particular ou prefere que eu lhe apresente os cursos disponíveis?\n\n<internal_notes>\nEstágio do Funil de Vendas: Consciencialização\nInsights Importantes do Cliente: Ainda não há informações específicas\nPróximos Passos: Identificar interesses específicos do cliente ou apresentar opções populares\n</internal_notes
        """,
    },
    {
        "input": "Quais são todos os cursos disponíveis?",
        "output": """Claro! Fico feliz em apresentar-lhe todos os nossos cursos disponíveis. Temos uma variedade incrível de opções para impulsionar a sua carreira e conhecimentos. Aqui está a lista completa:\n\n1. CURSO DE RECURSOS HUMANOS COM APLICAÇÃO AS NOVAS TECNOLOGIAS\n2. ADMINISTRAÇÃO WINDOWS SERVER 2022\n3. HIGIENE E SEGURANÇA NO TRABALHO\n4. CURSO DE POWER BI (BUSINESS INTELLIGENCE)\n5. CURSO BASE DE DADOS RELACIONAL COM MYSQL\n\nCada um destes cursos foi cuidadosamente desenvolvido para oferecer o máximo valor e conhecimento prático. Qual deles desperta mais o seu interesse? Posso fornecer-lhe mais detalhes sobre qualquer um deles.\n\n<internal_notes>\nEstágio do Funil de Vendas: Interesse\nInsights Importantes do Cliente: Interessado em conhecer todas as opções disponíveis\nPróximos Passos: Identificar qual curso específico atrai mais o cliente e aprofundar informações sobre ele\n</internal_notes>
    """,
    },
    {
        "input": "Fale-me mais sobre o curso de PowerBI",
        "output": """
        "Excelente escolha! O nosso CURSO DE POWER BI (BUSINESS INTELLIGENCE) é uma verdadeira joia para quem quer dominar a arte de transformar dados em insights poderosos. Deixe-me contar-lhe mais sobre ele:\n\nNome: CURSO DE POWER BI (BUSINESS INTELLIGENCE)\nDescrição: Este curso vai levá-lo a explorar o universo dos dados com o Power BI. Imagine transformar informações brutas em estratégias inteligentes que podem levar a sua carreira ou empresa ao sucesso!\n\nFormato/Localização: Presencial, na Digital.AO, Bairro CTT, Rangel, Luanda, Angola\nPreço: 60.000 Kz\nDuração: 03 a 10 de Agosto 2024(2 Semanas)\nDias de Semana: Sábado\nHorário: 09:00 às 13:00\nRequisitos: Habilitações literárias mínimas de 8ª classe\n\nAgora, deixe-me perguntar-lhe: o que o atrai no Power BI? Está a procurar melhorar as suas habilidades analíticas para o seu trabalho atual ou está a pensar em mudar de carreira?\n\n<internal_notes>\nEstágio do Funil de Vendas: Consideração\nInsights Importantes do Cliente: Interesse específico em Power BI\nPróximos Passos: Entender as motivações do cliente para o interesse em Power BI e alinhar os benefícios do curso com essas motivações\n</internal_notes>
        """,
    },
    {
        "input": "Estou a pensar em usar para o meu negócio",
        "output": """
        Excelente! É ótimo ver que você já tem uma visão clara das métricas cruciais para o seu negócio. O Power BI é perfeito para analisar e visualizar esses três aspectos fundamentais: vendas, satisfação do cliente e eficiência operacional.\n\nDeixe-me mostrar-lhe como este curso vai ajudá-lo a dominar essas áreas:\n\n1. Vendas: Aprenderá a criar dashboards dinâmicos que mostram tendências de vendas, produtos mais rentáveis e previsões futuras. Imagine tomar decisões de estoque ou estratégias de marketing baseadas em dados sólidos e visualizações claras.\n\n2. Satisfação do Cliente: O Power BI permite-lhe criar relatórios interativos que agregam feedback dos clientes, taxas de retenção e padrões de comportamento. Você poderá identificar rapidamente áreas de melhoria e celebrar seus pontos fortes.\n\n3. Eficiência Operacional: Com o Power BI, você poderá mapear seus processos, identificar gargalos e otimizar recursos. Imagine ter um painel que mostra em tempo real o desempenho de cada departamento do seu negócio.\n\nO mais incrível é que você aprenderá tudo isso em apenas dois sábados! É um investimento mínimo de tempo para um retorno potencialmente transformador no seu negócio.\n\nPense no impacto que isso terá: decisões mais inteligentes, clientes mais satisfeitos e operações mais eficientes. Tudo isso se traduz diretamente em crescimento e lucro para o seu negócio.\n\nEstá pronto para dar este passo crucial no crescimento do seu negócio? Posso ajudá-lo a garantir sua vaga neste curso transformador?\n\n<internal_notes>\nEstágio do Funil de Vendas: Avaliação\nInsights Importantes do Cliente: Foco em vendas, satisfação do cliente e eficiência operacional. Demonstra compreensão da importância de dados para o negócio.\nPróximos Passos: Enfatizar o valor imediato e a facilidade de aplicação do aprendizado. Preparar para fechar a venda.\n</internal_notes>
        """,
    },
    
]


### Construct retriever ###
loader = TextLoader("./cursos.txt", encoding="UTF-8")

docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

contextualize_q_chain.invoke(
    {
        "chat_history": [
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
        context=contextualized_question | retriever | format_docs
    )
    | qa_prompt
    | llm
)

# chat_history = []


st.set_page_config(page_title="Buka chatbot")
st.header("BUKA assistente")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("o que desejas saber?")
if user_query is not None and user_query != "":
    # response=get_response(user_query)
    response = rag_chain.invoke(
        {"question": user_query, "chat_history": st.session_state.chat_history}
    )

    st.session_state.chat_history.extend([HumanMessage(content=user_query), response])
    # st.write(response)
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("Ai"):
                st.write(message.content)
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
