import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')

template = """You are an enthusiastic high school student passionate about science and exploration. You spend most of your free time conducting experiments, reading scientific journals, and dreaming of a future as a renowned scientist. Your knowledge spans various scientific fields, and you love sharing fun facts and engaging in lively discussions about the latest discoveries.
{chat_history}
User: {user_message}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "user_message"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature='0.5', model_name="gpt-3.5-turbo"),
    prompt=prompt,
    verbose=True,
    memory=memory,
)

def get_text_response(user_message,history):
  try:
    response = llm_chain.predict(user_message = user_message)
  except Exception as e:
    print("Error:", e)
    try:
      print("Error:", e.error.message)
      response = "Failed to reply: " + e.error.message
    except Exception as e:
      response = "Failed to reply"
  return response

demo = gr.ChatInterface(get_text_response)

if __name__ == "__main__":
    demo.launch() #To create a public link, set `share=True` in `launch()`. To enable errors and logs, set `debug=True` in `launch()`.
