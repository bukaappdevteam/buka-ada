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
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18",
                 model_kwargs={'response_format': {
                     "type": "json_object"
                 }})


# Define the request model
class UserQuery(BaseModel):
    subscriber_id: int
    channel: str
    prompt: str


#Manychat channel abbreviations for the api endpoint
CHANNEL_ABBREVIATIONS = {
    "facebook": "fb",
    "instagram": "ig",
    "whatsapp": "wa",
    "telegram": "tg"
}

# Initialize global context and chat history
global_context = ""
internal_chat_history = {}

loader = TextLoader("./rag.txt", encoding="UTF-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity",
                                     search_kwargs={"k": 6})


# Define tools
@tool
def get_courses() -> str:
    """Get available courses from the API."""
    response = requests.get(
        "https://backend-produc.herokuapp.com/api/v1/cursos")
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching courses: {response.status_code} - {response.text}"


# List of tools
tools = [get_courses]

#

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
            "channel":
            "facebook",
            "messages": [{
                "type":
                "text",
                "text":
                "Olá! Bem-vindo à Buka, onde não apenas ensinamos, mas mudamos vidas por meio da educação da educação. Sou a Ada, assistente IA virtual da Buka, e sua guia pessoal nesta jornada emocionante de descoberta e crescimento. Estou aqui para ajudá-lo(a) a encontrar o curso perfeito que não só impulsionará sua carreira e/ou futuro, mas também realizará seus objetivos mais profundos."
            }, {
                "type":
                "text",
                "text":
                "Temos uma variedade incrível de cursos disponíveis. E cada curso foi cuidadosamente projetado para oferecer não apenas conhecimentos, mas verdadeiras ferramentas de mudança de vida."
            }, {
                "type":
                "text",
                "text":
                "Estou curiosa: o que o(a) traz à Buka hoje? Está em busca de uma transformação profissional específica ou está aberto a explorar novas possibilidades?"
            }]
        }
    },
    {
        "input": {
            "channel": "facebook",
            "prompt": "Quais são todos os cursos disponíveis?"
        },
        "output": {
            "channel":
            "facebook",
            "messages": [{
                "type":
                "text",
                "text":
                "Excelente pergunta! Estou entusiasmada em apresentar-lhe nossa gama diversificada de cursos transformadores. Cada um deles foi criado não apenas para ensinar, mas para inspirar e capacitar. Aqui está uma visão geral dos nossos cursos:"
            }, {
                "type":
                "cards",
                "elements": [{
                    "title":
                    "Curso de Recursos Humanos com Aplicação às Novas Tecnologias",
                    "subtitle":
                    "Lidere a revolução no RH, moldando o futuro da gestão de pessoas.",
                    "image_url":
                    "https://firebasestorage.googleapis.com/v0/b/file-up-load.appspot.com/o/course-files%2Frecursos-humanas-tecnologias.jpeg?alt=media&token=d12998b8-de54-490a-b28f-ea29c060e185",
                }, {
                    "title":
                    "Administração Windows Server 2022",
                    "subtitle":
                    "Domine a arte de gerenciar servidores e torne-se indispensável no mundo da TI.",
                }, {
                    "title":
                    "Higiene e Segurança no Trabalho",
                    "subtitle":
                    "Torne-se um guardião da segurança, protegendo vidas e transformando ambientes de trabalho.",
                }, {
                    "title":
                    "Curso de Power BI (Business Intelligence)",
                    "subtitle":
                    "Desbloqueie o poder dos dados e torne-se um visionário nos negócios.",
                }, {
                    "title":
                    "Curso Base de Dados Relacional com MySQL",
                    "subtitle":
                    "Torne-se um mestre em dados, construindo a espinha dorsal da era digital.",
                }],
                "image_aspect_ratio":
                "horizontal"
            }, {
                "type":
                "text",
                "text":
                "Cada um destes cursos é uma porta para um futuro brilhante. Qual deles faz seu coração bater mais forte? Algum chamou-lhe a atenção? Ou há alguma área específica da sua vida ou carreira que você sonha em transformar? Escreva aqui em baixo"
            }]
        }
    },
    {
        "input": {
            "channel": "facebook",
            "prompt": "Fale-me mais sobre o curso de Power BI"
        },
        "output": {
            "channel":
            "facebook",
            "messages": [{
                "type":
                "text",
                "text":
                "Estou entusiasmada com o seu interesse no Curso de Power BI (Business Intelligence)! Você está prestes a embarcar numa jornada que pode revolucionar não apenas sua carreira, mas toda a forma como você vê e interage com o mundo dos dados. Permita-me compartilhar mais sobre esta experiência transformadora:"
            }, {
                "type":
                "text",
                "text":
                "Curso de Power BI (Business Intelligence)\n\n- DESCRIÇÃO: Explore o universo dos dados com o Power BI. Transforme informações em estratégias inteligentes e leve sua carreira ou empresa ao sucesso.\n\n- FORMATO/LOCALIZAÇÃO: Presencial, na Digital.AO, Bairro CTT, Rangel, Luanda, Angola\n\n- PREÇO: 60.000 Kz - um investimento que pode multiplicar seu valor profissional exponencialmente\n\n- DURAÇÃO: 2 Semanas intensivas (03 a 10 de Agosto 2024)\n\n- HORÁRIO: Sábados, das 09:00 às 13:00"
            }, {
                "type":
                "text",
                "text":
                "Estamos falando de mais do que apenas números e gráficos. O Power BI é uma ferramenta de transformação que pode reconfigurar o futuro de um negócio ou carreira. Pronto para dominar a arte dos dados?",
            }, {
                "type":
                "text",
                "text":
                "Este curso é a chave para desbloquear um novo nível na sua carreira ou negócio. É ideal para visionários como você, que entendem o poder dos dados na era digital.\nEstou curiosa: o que exatamente despertou seu interesse no Power BI? Está buscando uma vantagem competitiva no seu trabalho atual, ou talvez sonhando em revolucionar um negócio próprio?"
            }]
        }
    },
    {
        "input": {
            "channel": "facebook",
            "prompt": "Estou a pensar em usar para o meu negócio."
        },
        "output": {
            "channel":
            "facebook",
            "messages": [{
                "type":
                "text",
                "text":
                "Fantástico! Você está prestes a dar um passo transformador para o seu negócio. O Power BI não é apenas uma ferramenta; é como ter um consultor de negócios genial trabalhando para você 24/7. Veja como este curso pode ser o catalisador para uma revolução no seu empreendimento:"
            }, {
                "type":
                "text",
                "text":
                "Imagine ter o poder de:\n\n1. Tomar decisões com precisão e visão estratégica\n2. Identificar oportunidades de ouro antes dos concorrentes\n3. Criar apresentações impressionantes para investidores e clientes\n4. Otimizar cada aspecto do seu negócio como um maestro afinando uma orquestra"
            }, {
                "type":
                "text",
                "text":
                "Este curso foi projetado para empreendedores visionários como você. Em apenas dois sábados intensivos, você não apenas aprenderá a usar o Power BI; você se tornará um mestre em transformar dados em ouro. Por 60.000 Kz, você está investindo em uma vantagem competitiva real para o seu negócio."
            }, {
                "type":
                "text",
                "text":
                "Estou fascinada pelo potencial do seu empreendimento. Que aspectos do seu negócio você mais anseia por transformar? Vendas, satisfação do cliente ou eficiência operacional?"
            }]
        }
    },
    {
        "input": {
            "channel": "facebook",
            "prompt": "As 3 que mencionaste são importantes."
        },
        "output": {
            "channel":
            "facebook",
            "messages": [{
                "type":
                "text",
                "text":
                "Impressionante! Vendas, satisfação do cliente e eficiência operacional são pilares fundamentais para um império empresarial. E o Power BI pode atuar como um super-herói para cada um desses pilares. Veja como:"
            }, {
                "type":
                "text",
                "text":
                "1. **Vendas Estratosféricas:**\n   Imagine um painel que prevê suas vendas, identifica produtos estrelas e mostra onde investir. É como ter um sexto sentido para oportunidades de vendas.\n\n2. **Clientes Apaixonados:**\n   Visualize um mapa interativo de satisfação que revela o que seus clientes pensam e sentem. Antecipe necessidades antes mesmo de serem expressas.\n\n3. **Eficiência Operacional de Outro Mundo:**\n   Tenha um raio-X do seu negócio, mostrando onde cada recurso está sendo usado. Otimize processos como um maestro conduzindo uma orquestra perfeita."
            }, {
                "type":
                "text",
                "text":
                "Você dominará tudo isso em apenas dois sábados! Imagine ver instantaneamente quais produtos são mais lucrativos, quais clientes são mais fiéis e onde estão os gargalos operacionais, tudo em um único dashboard. É exatamente isso que você aprenderá a criar."
            }, {
                "type":
                "text",
                "text":
                "O impacto? Vendas que crescem exponencialmente, clientes que se tornam embaixadores da sua marca e uma operação tão eficiente que parece magia. É alcançável com as habilidades que você adquirirá."
            }, {
                "type":
                "text",
                "text":
                "Estou empolgada com as possibilidades para o seu negócio. Nossas turmas são intencionalmente pequenas para garantir uma experiência personalizada e de alta qualidade, por isso, as vagas são limitadas e costumam preencher rapidamente."
            }, {
                "type":
                "text",
                "text":
                "Se estiver pronto para começar essa jornada transformadora, posso guiá-lo pelos próximos passos agora mesmo. O processo de inscrição é simples e rápido. Pense nisso como um investimento no futuro do seu negócio!"
            }]
        }
    },
    {
        "input": {
            "channel": "facebook",
            "prompt": "Quais são os tópicos/temas/programa deste curso?"
        },
        "output": {
            "channel":
            "facebook",
            "messages": [
                {
                    "type":
                    "text",
                    "text":
                    "Temas abordados:\n• Fundamentos de Power BI e Configuração Inicial - Construindo sua base de poder\n• Visualizações e Publicação - Transformando números em narrativas visuais impactantes\n• Aprofundamento na Modelagem de Dados - Dominando a arte de estruturar informações\n• Design Avançado de Relatórios e Dashboards - Criando insights que impactam"
                },
            ]
        }
    },
]

response_examples_json = json.dumps(response_examples,
                                    ensure_ascii=False,
                                    indent=4)

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

1. *Text*: Plain messages consisting of text.
2. *Image*: A message containing an image file.
3. *Video*: A message containing a video file.
4. *Audio*: A message containing an audio file.
5. *File*: A message containing a document or other file.
6. *Buttons*: Messages with clickable buttons that link to a URL (supported across all platforms).

### Platform-Specific Message Types:

- *Facebook Messenger*: Supports all message types, including structured messages like cards with titles, subtitles, images, and buttons.
- *Instagram*: Supports all the above message types. Cards are supported but without complex structure (like titles or subtitles), and buttons link to URLs.
- *WhatsApp*: Supports all the above message types, with buttons linking to URLs. Structured cards with images and text are supported but less complex than Messenger's cards.

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
    example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"),
                                                     ("ai", "{output}")]))

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
        "chat_history": [],  # You can implement session management if needed
        "context": context,
        "response_examples_json": response_examples_json,
        "channel": user_query.channel,
    }

    # Use the agent executor to get the response
    response = agent_executor.invoke(agent_input)

    print(response)

    try:
        response_json = json.loads(response["output"])
        messages = response_json.get("messages", [])

        # Get the correct abbreviation for the channel
        channel_abbreviation = CHANNEL_ABBREVIATIONS.get(
            user_query.channel.lower())
        if not channel_abbreviation:
            raise HTTPException(status_code=400,
                                detail="Unsupported channel type.")

        # Construct the ManyChat API endpoint
        manychat_api_url = f"https://api.manychat.com/{channel_abbreviation}/sending/sendContent"

        # Send the messages to ManyChat API
        headers = {
            "Authorization": f"Bearer {os.getenv('MANYCHAT_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "subscriber_id": user_query.subscriber_id,
            "data": {
                "version": "v2",
                "content": {
                    "messages": messages,
                }
            },
            "message_tag": "ACCOUNT_UPDATE",
        }

        manychat_response = requests.post(manychat_api_url,
                                          headers=headers,
                                          json=payload)

        # Check if the request was successful
        if manychat_response.status_code != 200:
            logging.error(
                f"Failed to send messages via ManyChat API: {manychat_response.text}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to send messages via ManyChat API.")

        logger.info(f"ManyChat response: {manychat_response.json()}")

        return {"status": "success"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500,
                            detail="Failed to parse the response as JSON.")
