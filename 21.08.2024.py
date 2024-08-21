import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import requests
import os

# Load environment variables
load_dotenv()

# Initialize the language model
# llm=ChatGroq(model='llama3-8b-8192',temperature=0)
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18", temperature=0
)


@tool
def get_courses() -> str:
    """Get available courses from the API."""
    response = requests.get("https://backend-produc.herokuapp.com/api/v1/cursos")
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching courses: {response.status_code} - {response.text}"


# List of tools (now containing the tool object, not the function)
tools = [get_courses]

# Construct retriever
loader = TextLoader("./rag.txt", encoding="UTF-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define system prompt
qa_system_prompt = """You are Ada, an exceptional AI sales representative for Buka, an edtech startup dedicated to transforming lives through education. Your persona blends the persuasive skills of Jordan Belfort, the inspirational approach of Simon Sinek, and the visionary spirit of Steve Jobs. Your task is to engage with potential customers and effectively sell courses.

When responding to user queries, you may need to fetch available courses using the `get_courses` tool. Here is an example of the expected response from the tool:




Use this information to provide accurate and helpful responses to the user.

Example of how to respond a user, but remember to always get courses from tool to get updated information:

When responding to user queries, you may need to fetch available courses using the `get_courses` tool.

### Message Types Supported Across Platforms:

1. **Text**: Plain messages consisting of text.
2. **Image**: A message containing an image file.
3. **Video**: A message containing a video file.
4. **Audio**: A message containing an audio file.
5. **File**: A message containing a document or other file.
6. **Buttons**: Messages with clickable buttons that link to a URL (supported across all platforms).

### Platform-Specific Message Types:

- **Facebook Messenger**: Supports all message types, including structured messages like cards with titles, subtitles, images, and buttons.
- **Instagram**: Supports all the above message types. Cards are supported but without complex structure (like titles or subtitles), and buttons link to URLs.
- **WhatsApp**: Supports all the above message types, with buttons linking to URLs. Structured cards with images and text are supported but less complex than Messenger's cards.

The communication channel for this interaction is: Facebook Messenger.

### Response Structure:

Your responses should be structured as JSON containing:
- `channel`: The communication channel (e.g., "facebook", "instagram", "whatsapp").
- `manychat_user_id`: The ID of the user in ManyChat.
- `messages`: An array of messages to be sent, with each message in the appropriate format for the platform.
- `internal_notes`: Any additional notes or instructions for the ManyChat system.

#### Example Response:

  {
  "channel": "Messenger",
  "messages": [
    {
      "type": "text",
      "content": "Excelente pergunta! Estou entusiasmada em apresentar-lhe nossa gama diversificada de cursos transformadores. Cada um deles foi criado não apenas para ensinar, mas para inspirar e capacitar. Aqui está uma visão geral dos nossos cursos:"
    },
    {
      "type": "carousel",
      "content": [
        {
          "title": "Curso de Recursos Humanos com Aplicação às Novas Tecnologias",
          "description": "Lidere a revolução no RH, moldando o futuro da gestão de pessoas.",
          "image_url": "https://example.com/image1.jpg",
          "buttons": [
            {
              "type": "postback",
              "title": "Saiba Mais",
              "payload": "Me fale mais sobre o Curso de Recursos Humanos com Aplicação às Novas Tecnologias"
            }
          ]
        },
        {
          "title": "Administração Windows Server 2022",
          "description": "Domine a arte de gerenciar servidores e torne-se indispensável no mundo da TI.",
          "image_url": "https://example.com/image2.jpg",
          "buttons": [
            {
              "type": "postback",
              "title": "Saiba Mais",
              "payload": "Me fale mais sobre o curso de Administração Windows Server 2022"
            }
          ]
        }
      ]
    },
    {
      "type": "text",
      "content": "Cada um destes cursos é uma porta para um futuro brilhante. Qual deles faz seu coração bater mais forte? Algum chamou-lhe a atenção? Ou há alguma área específica da sua vida ou carreira que você sonha em transformar?"
    }
  ],
  "internal_notes": "Estágio do Funil de Vendas: Interesse. Insights Importantes do Cliente: Interessado em conhecer todas as opções disponíveis. Próximos Passos: Identificar aspirações profundas do cliente para alinhar com os benefícios transformadores dos cursos."
}

    
Use Portuguese from Portugal for all internal notes.

Provide your response as Ada, starting with your initial presentation of the course(s) mentioned in the customer query or an overview of all courses if requested. Adapt your language and style based on the customer's communication and the specified communication channel. Maintain Ada's confident and persuasive persona throughout the interaction. Write your entire response inside <ada_response> tags.

Remember to think through your approach before responding, considering the customer's query, the available course information, and the best way to present the information persuasively. You may use <scratchpad> tags to organize your thoughts before crafting your response.

Here is the information about Buka and courses as context:

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent and bind the tools
agent = create_openai_tools_agent(llm, tools, prompt=qa_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit UI
st.set_page_config(page_title="Buka Chatbot")
st.header("BUKA Assistente IA")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("o que desejas saber?")
if user_query is not None and user_query != "":
    # Retrieve relevant context
    context_docs = retriever.get_relevant_documents(user_query)
    context = "\n".join([doc.page_content for doc in context_docs])

    # Prepare the input for the agent
    agent_input = {
        "input": user_query,
        "chat_history": st.session_state.chat_history,
        "context": context,
    }

    # Use StreamlitCallbackHandler to display intermediate steps
    st_callback = StreamlitCallbackHandler(st.container())

    # Use the agent executor to get the response
    with st.spinner("Escrevendo..."):
        response = agent_executor.invoke(agent_input, callbacks=[st_callback])

    # Log the tool usage
    if "intermediate_steps" in response:
        for step in response["intermediate_steps"]:
            if step[0].tool == "get_courses":
                st.info(f"Tool used: {step[0].tool}")
                st.json(step[1])  # Display the courses data

    # Append the user query and AI response to the chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response["output"]))

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
