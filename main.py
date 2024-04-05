# We will integrate our code with OpenAI
import os

from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key
 
#Streamlit framework

st.title('Get your celebrity gossip :P')
input_text = st.text_input("Enter the name of your favourite celebrity, My dear friend :)")
llm = OpenAI(temperature = 0.7)
#Prompt templates
first_input_prompt_template = PromptTemplate(
    input_variables = ['name'],
    template ="Tell me the celebrity {name}"
)

chain = LLMChain(llm=llm, prompt = first_input_prompt_template, verbose = True, output_key='person')

second_input_prompt_template = PromptTemplate(
    input_variables = ['person']
    template ="Tell the {person}'s horoscope"
)
chain2 = LLMChain(llm=llm, prompt = second_input_prompt_template, verbose = True, output_key='horoscope')

third_input_prompt_template = PromptTemplate(
    input_variables = ['horoscope']
    template ="List 5 celebrities who have the same {horoscope}"
)
chain3 = LLMChain(llm=llm, prompt = third_input_prompt_template, verbose = True, output_key='persons')



person_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
horoscope_memory = ConversationBufferMemory(input_key='horoscope', memory_key='chat_history')
persons_memory = ConversationBufferMemory(input_key='persons', memory_key='description_history')


main_chain = SequentialChain(chains= [chain, chain2, chain3], input_variables = ['name'], output_variables = ['person', 'horoscope', 'persons'], verbose=True)


if input_text:
    st.write(main_chain.run({'name':input_text}))