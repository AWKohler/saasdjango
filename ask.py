from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import os

os.environ["OPENAI_API_KEY"] = "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"

embeddings = OpenAIEmbeddings()

# docsearch = Chroma(persist_directory="db", embedding_function=embeddings)
docsearch = Chroma(persist_directory="media/Johnson/db", embedding_function=embeddings)

chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.as_retriever())
# chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=docsearch.similarity_search())

user_input = input("What's your question: ")

result = chain({"question": user_input}, return_only_outputs=True)

print("Answer: " + result["answer"].replace('\n', ' '))
print("Source: " + result["sources"])
print(result)
