import streamlit as st
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.prompts.chat import ChatPromptTemplate
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from openai import OpenAI
from utils import record_audio, play_audio
import datetime
import warnings
import pygame

# Ignore DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database connection
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
        Give an answer if asked about any products in the database by querying description, offers, and price for the product.
        Search products in every column, since there may not be any category with any item.
        Please use the below context to write the SQL queries. It is a PostgreSQL database.
        Dont return the final answer in dictionary format. Use string format.
        Give response in a concise paragraph.
        Context:
        You are an expert at handling databases.
        You must query against the connected database, which has tables 'products', 'products_russian'.
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
    handle_parsing_errors=True,  # Ensuring parsing errors are handled
)

# Function to get the final answer
def get_final_answer(question):
    try:
        output = agent.invoke(prompt.format_prompt(question=question))
        # Print the raw output for debugging
        st.write(f"Raw Output: {output}")
        
        # Extract only the final answer from the nested dictionary
        if isinstance(output, dict) and 'output' in output:
            output_text = output['output']
            # Ensure the output is correctly formatted
            if "Final Answer:" in output_text:
                final_answer = output_text.split("Final Answer:")[-1].strip()
            else:
                final_answer = output_text.strip()
        else:
            final_answer = str(output)
        
        # Generate audio for the final answer
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=final_answer)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"output_{timestamp}.mp3"
        response.stream_to_file(filename)

        # Debugging: Ensure the file exists
        st.write(f"Audio file created: {filename}")
        if not os.path.exists(filename):
            st.error("Audio file was not created.")
            return "Audio file was not created."

        # Play the audio file
        play_audio(filename)
        return final_answer
        
    except Exception as e:
        return f"An error occurred: {e}"

# Main function to create Streamlit UI
def main():
    st.title("Chat with DB")

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    def display_messages():
        for msg in st.session_state['messages']:
            if msg['role'] == 'user':
                st.chat_message("user").markdown(msg['content'])
            else:
                st.chat_message("assistant").markdown(msg['content'])

    # Chat input section
    st.header("Chat Input")

    # Add an audio recording button
    if st.button("🎤"):
        try:
            if sr.Microphone().list_microphone_names():
                with st.spinner("Recording..."):
                    record_audio('test.wav')
                    st.success("Audio recorded successfully.")
                audio_file = open('test.wav', "rb")
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                question = transcription.text
                st.session_state['messages'].append({'role': 'user', 'content': question})
                with st.spinner("Processing..."):
                    final_answer = get_final_answer(question)
                st.session_state['messages'].append({'role': 'assistant', 'content': final_answer})
            else:
                st.error("No microphone found. Please connect a microphone and try again.")
        except OSError as e:
            st.error(f"An error occurred: {e}")

    # Text input section
    question = st.chat_input("Enter your question:")
    if question:
        st.session_state['messages'].append({'role': 'user', 'content': question})
        with st.spinner("Processing..."):
            final_answer = get_final_answer(question)
        st.session_state['messages'].append({'role': 'assistant', 'content': final_answer})
        display_messages()

if __name__ == "__main__":
    main()
