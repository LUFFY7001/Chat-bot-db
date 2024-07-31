import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
import wave
import openai
import pyaudio
from sqlalchemy import create_engine
from dotenv import load_dotenv
import time
import os
import warnings

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate

# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up database connection
cs = os.getenv("CON_KEY")
db_engine = create_engine(cs)
db = SQLDatabase(db_engine)

# Initialize the language model
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.0, model="gpt-4o-mini")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_toolkit.get_tools()

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        """
        You are a sales agent and you know a lot about marketing.
        You have to give short and concise responses with kindness.
        Your products are in the 'products' database.
        Reply the final answer as a human would.
        If the user input is in Russian script then reply the final answer in Russian script as well.
        If the user input is in Russian script then use products_russian table.
        Product names may be case sensitive (consider the possibility).
        If finding difficulty in finding specifications, search for individual words in the database.
        Give an answer if asked about any products in the database by querying description, offers and price for the product.
        Search products in every column. since there may not be any category with any item.
        Please use the below context to write the SQL queries. It is a PostgreSQL database.
        Context:
        You are an expert at handling databases.
        You must query against the connected database, which has tables, 'products', 'products_russian'.
        Both tables have columns: name, description, price, category, offers, image_link.
        As an expert, you must use joins whenever required.
        """
        ),
        ("human", "{question}\nAI:")
    ]
)

# Create the SQL agent
agent = create_sql_agent(
    llm=llm,
    toolkit=sql_toolkit,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_execution_time=100,
    max_iterations=1000,
    handle_parsing_errors=True
)

# Function to convert speech to text
def speech_to_text(audio_file):
    transcription = client.audio.transcriptions.create(model="whisper-1", file=open(audio_file, "rb"))
    return transcription.text

# Function to convert text to speech and play it
def text_to_speech_and_save(response_text):
    wave_file_path = f"tts_output_{int(time.time())}.wav"
    with wave.open(wave_file_path, 'wb') as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)  # Assume 16-bit PCM
        wave_file.setframerate(24000)
        # Simulate TTS data generation (replace this with your actual TTS data)
        for _ in range(100):
            wave_file.writeframes(b'\x00\x00' * 24000)  # Replace with actual TTS data
    return wave_file_path

# Function to get the final answer
def get_final_answer(question):
    try:
        output = agent.invoke(prompt.format_prompt(question=question))
        if isinstance(output, dict) and 'output' in output:
            output_text = output['output']
            if "Final Answer:" in output_text:
                final_answer = output_text.split("Final Answer:")[-1].strip()
            else:
                final_answer = output_text.strip()
        else:
            final_answer = str(output)

        wave_file_path = text_to_speech_and_save(final_answer)
        return wave_file_path, final_answer

    except Exception as e:
        return None, f"An error occurred: {e}"

# Streamlit UI
st.title("Chat with Database")

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

input_option = st.radio("Choose input type:", ("Text", "Audio"))

if input_option == "Text":
    question = st.chat_input("Enter your question:")
    if question:
        wave_file_path, final_answer = get_final_answer(question)
        st.session_state.chat_history.append({"question": question, "answer": final_answer})
        st.write("Answer:", final_answer)
        if wave_file_path:
            st.audio(wav_file_path)

elif input_option == "Audio":
    audio_bytes = audio_recorder()
    if audio_bytes:
        # Save the audio file
        wavfile = f"realtime_audio_{int(time.time())}.wav"
        with open(wavfile, "wb") as f:
            f.write(audio_bytes)
        
        st.write("Audio recorded. Processing...")
        question = speech_to_text(wavfile)
        st.write("Transcribed Question:", question)
        wave_file_path, final_answer = get_final_answer(question)
        st.session_state.chat_history.append({"question": question, "answer": final_answer})
        st.write("Answer:", final_answer)
        if wave_file_path:
            st.audio(wav_file_path)

# Display the chat history
st.write("### Chat History")
for i, chat in enumerate(st.session_state.chat_history):
    st.write(f"**Q{i+1}:** {chat['question']}")
    st.write(f"**A{i+1}:** {chat['answer']}")
