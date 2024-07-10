import streamlit as st
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
st.header('Assistente Buka')
prompt=st.text_input(label="Escreva aqui")
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    #max_tokens=1024,
    #timeout=None,
    #max_retries=2,
    # other params...
)
if prompt:
    msg=llm.invoke(prompt)
    st.write(msg.content)