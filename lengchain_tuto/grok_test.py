from langchain_xai import ChatXAI

# Initialize Grok model
llm = ChatXAI(
    model="grok-beta",  # or another Grok model name, check xAI docs for options
    xai_api_key="your-api-key-here"
)

# Invoke a simple query
response = llm.invoke("Tell me about the universe.")
print(response.content)


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm
response = chain.invoke({"input": "Explain black holes."})
print(response.content)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# First question
print(conversation.run("What is AI?"))

# Follow-up question
print(conversation.run("How does it relate to me?"))