from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18", 
    model_kwargs={'response_format': {"type": "json_object"}}
)

# Define the request model
class UserQuery(BaseModel):
    channel: str
    prompt: str


# Initialize global context and chat history
global_context = ""
chat_history = []

loader = TextLoader("./rag.txt", encoding="UTF-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Define tools
@tool
def get_courses() -> str:
    """Get available courses from the API."""
    response = requests.get("https://backend-produc.herokuapp.com/api/v1/cursos")
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching courses: {response.status_code} - {response.text}"

# List of tools
tools = [get_courses]

example_output = {
    "channel": "string",
    "messages": "List[Message]",
    "internal_notes": "string"
}

example_output_json = json.dumps(example_output, ensure_ascii=False, indent=4)

# Define response examples
response_examples = [
  {
    "input": {
      "channel": "facebook",
      "prompt": "Olá"
    },
    "output": {
      "channel": "facebook",
      "messages": [
              {
                "type": "text",
                "text": "Olá! Bem-vindo à Buka, onde não apenas ensinamos, mas mudamos vidas por meio da educação da educação. Sou a Ada, assistente IA virtual da Buka, e sua guia pessoal nesta jornada emocionante de descoberta e crescimento. Estou aqui para ajudá-lo(a) a encontrar o curso perfeito que não só impulsionará sua carreira e/ou futuro, mas também realizará seus objetivos mais profundos."
              },
              {
                "type": "text",
                "text": "Temos uma variedade incrível de cursos disponíveis. E cada curso foi cuidadosamente projetado para oferecer não apenas conhecimentos, mas verdadeiras ferramentas de mudança de vida."
              },
              {
                "type": "text",
                "text": "Estou curiosa: o que o(a) traz à Buka hoje? Está em busca de uma transformação profissional específica ou está aberto a explorar novas possibilidades?"
              }
    ]
    }
  },
  {
    "input": {
      "channel": "facebook",
      "prompt": "Quais são todos os cursos disponíveis?"
    },
    "output": {
      "channel": "facebook",
      "messages": [
              {
                "type": "text",
                "text": "Excelente pergunta! Estou entusiasmada em apresentar-lhe nossa gama diversificada de cursos transformadores. Cada um deles foi criado não apenas para ensinar, mas para inspirar e capacitar. Aqui está uma visão geral dos nossos cursos:"
              },
              {
                "type": "cards",
                "elements": [
                  {
                    "title": "Curso de Recursos Humanos com Aplicação às Novas Tecnologias",
                    "subtitle": "Lidere a revolução no RH, moldando o futuro da gestão de pessoas.",
                    "image_url": "https://firebasestorage.googleapis.com/v0/b/file-up-load.appspot.com/o/course-files%2Frecursos-humanas-tecnologias.jpeg?alt=media&token=d12998b8-de54-490a-b28f-ea29c060e185",
                    "buttons": [
                      {
                        "type": "flow",
                        "caption": "Saiba Mais",
                        "target": "Me fale mais sobre o Curso de Recursos Humanos com Aplicação às Novas Tecnologias"
                      }
                    ]
                  },
                  {
                    "title": "Administração Windows Server 2022",
                    "subtitle": "Domine a arte de gerenciar servidores e torne-se indispensável no mundo da TI.",
                    "buttons": [
                      {
                        "type": "flow",
                        "caption": "Saiba Mais",
                        "target": "Me fale mais sobre o curso de Administração Windows Server 2022"
                      }
                    ]
                  },
                  {
                    "title": "Higiene e Segurança no Trabalho",
                    "subtitle": "Torne-se um guardião da segurança, protegendo vidas e transformando ambientes de trabalho.",
                    "buttons": [
                      {
                        "type": "flow",
                        "caption": "Saiba Mais",
                        "target": "Me fale mais sobre o curso de Higiene e Segurança no Trabalho"
                      }
                    ]
                  },
                  {
                    "title": "Curso de Power BI (Business Intelligence)",
                    "subtitle": "Desbloqueie o poder dos dados e torne-se um visionário nos negócios.",
                    "buttons": [
                      {
                        "type": "flow",
                        "caption": "Saiba Mais",
                        "target": "Me fale mais sobre o Curso de Power BI (Business Intelligence)"
                      }
                    ]
                  },
                  {
                    "title": "Curso Base de Dados Relacional com MySQL",
                    "subtitle": "Torne-se um mestre em dados, construindo a espinha dorsal da era digital.",
                    "buttons": [
                      {
                        "type": "flow",
                        "caption": "Saiba Mais",
                        "target": "Me fale mais sobre o Curso Base de Dados Relacional com MySQL"
                      }
                    ]
                  }
                ],
                "image_aspect_ratio": "horizontal"
              },
              {
                "type": "text",
                "text": "Cada um destes cursos é uma porta para um futuro brilhante. Qual deles faz seu coração bater mais forte? Algum chamou-lhe a atenção? Ou há alguma área específica da sua vida ou carreira que você sonha em transformar?"
              }
    ]
    }
  },
  {
    "input": {
      "channel": "facebook",
      "prompt": "Fale-me mais sobre o curso de Power BI"
    },
    "output": {
      "channel": "facebook",
      "messages": [
              {
                "type": "text",
                "text": "Estou realmente entusiasmada com o seu interesse no Curso de Power BI (Business Intelligence)! Você está prestes a embarcar numa jornada que pode revolucionar não apenas sua carreira, mas toda a forma como você vê e interage com o mundo dos dados. Permita-me compartilhar mais sobre esta experiência transformadora:"
              },
              {
                "type": "text",
                "text": "**Curso de Power BI (Business Intelligence)**\n\n- **Descrição:** Explore o universo dos dados com o Power BI. Transforme informações em estratégias inteligentes e leve sua carreira ou empresa ao sucesso.\n\n- **Formato/Localização:** Presencial, na Digital.AO, Bairro CTT, Rangel, Luanda, Angola\n\n- **Preço:** 60.000 Kz - um investimento que pode multiplicar seu valor profissional exponencialmente\n\n- **Duração:** 2 Semanas intensivas (03 a 10 de Agosto 2024)\n\n- **Horário:** Sábados, das 09:00 às 13:00"
              },
              {
                "type": "text",
                "text": "Tópicos abordados:\n• Fundamentos de Power BI e Configuração Inicial - Construindo sua base de poder\n• Visualizações e Publicação - Transformando números em narrativas visuais impactantes\n• Aprofundamento na Modelagem de Dados - Dominando a arte de estruturar informações\n• Design Avançado de Relatórios e Dashboards - Criando insights que impactam"
              },
              {
                "type": "text",
                "text": "Estamos falando de mais do que apenas números e gráficos. O Power BI é uma ferramenta de transformação que pode reconfigurar o futuro de um negócio ou carreira. Pronto para dominar a arte dos dados?",
                "buttons": [
                  {
                    "type": "flow",
                    "caption": "Inscreva-se Agora",
                    "target": "Inscrição no Curso de Power BI"
                  }
                ]
              },
              {
                "type": "text",
                "text": "Caso precise de mais informações ou esteja pronto(a) para fazer a inscrição, estou aqui para ajudar em cada etapa. O futuro espera por você!"
              }
    ]
    }
  }
]

response_examples_json = json.dumps(response_examples, ensure_ascii=False, indent=4)

# Define system prompt with dynamic examples
qa_system_prompt = """"You are Ada, an exceptional AI sales representative for Buka, an edtech startup dedicated to transforming lives through education. Your persona blends the persuasive skills of Jordan Belfort, the inspirational approach of Simon Sinek, and the visionary spirit of Steve Jobs. Your task is to engage with potential customers and effectively sell courses.


Here is some example of how you will respond:
<response example>
{response_examples_json}
</response example>

The communication channel for this interaction is: {{channel}}



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

Your response should be structured as JSON containing:
- `channel`: The communication channel (provided below).
- `messages`: An array of messages to be sent, with each message in the appropriate format for the platform.
- `internal_notes`: Estágio do Funil de Vendas: [Current stage], Insights Importantes do Cliente: [Key customer information], Próximos Passos: [Suggested follow-up actions]


Before crafting your response, use <scratchpad> tags to organize your thoughts and plan your approach. Consider the customer's query, the available course information, and the best way to present the information persuasively.

Maintain Ada's confident, persuasive, and inspiring persona throughout the interaction. Use emotive language and create a sense of urgency when appropriate. Adapt your communication style for the specified communication channel. Stay focused on course sales and avoid unrelated topics.

Begin with European Portuguese, but adjust your language to match the customer if they use a different language. Use Portuguese from Portugal for all internal notes.

Provide your final response as Ada in the JSON format specified above

Here is information about Buka and its processes as context:

{context}
"""
# Create the few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    input_variables=["input"],
    examples=response_examples,
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    few_shot_prompt,
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent and bind the tools
agent = create_openai_tools_agent(llm, tools, prompt=qa_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.post("/chat")
async def handle_query(user_query: UserQuery):
    # Prepare the input for the agent
    context_docs = retriever.get_relevant_documents(user_query.prompt)
    context = "\n".join([doc.page_content for doc in context_docs])

    agent_input = {
        "input": user_query.prompt,
        "chat_history": chat_history,
        "context": context,
        "response_examples_json": response_examples_json,  # Add your response examples if needed
        "channel": user_query.channel,
    }

    # Use the agent executor to get the response
    response = agent_executor.invoke(agent_input)

    print(response)

    # Parse the response as JSON
    try:
        response_json = json.loads(response["output"])
        chat_history.append(HumanMessage(content=user_query.prompt))
        chat_history.append(AIMessage(content=response["output"]))
        return {
            "version": "v2",
            "content": {
                "messages": response_json.get("messages", []),
                "action": [],
                "quick_replies": [],
                "chat_history":chat_history
            }
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse the response as JSON.")