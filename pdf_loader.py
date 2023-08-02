from dotenv import load_dotenv
load_dotenv()

import os

from langchain.llms import OpenAI
from langchain.document_loaders import PDFMinerLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain import PromptTemplate,LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# import weaviate



loader = PDFMinerLoader("data.pdf")


documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

template = """Assistant, powered by OpenAI, is a sophisticated language model that has been trained utilizing data incorporated in the Vector Store.

The design of Assistant enables it to provide support in comprehending the information embedded in the data store, from answering straightforward questions to offering comprehensive explanations and engaging in discussions about a myriad of topics covered in the associated paper. As a language model, Assistant generates text that emulates human conversation based on the input it receives, facilitating natural-sounding dialogue and responses that are pertinent and cohesive to the topic of discussion.

In essence, Assistant serves as a versatile tool that can assist with a broad spectrum of tasks, delivering valuable insights and details about a wide range of topics outlined in the paper. Whether you require assistance with a specific inquiry or wish to engage in a conversation about a particular subject, Assistant is at your service.

In instances where Assistant is unable to provide an answer, it responds with 'The answer you're seeking is out of my memory.'

{context}

Human: {question}
Assistant:"""


_template = PromptTemplate(
    template=template,
    input_variables=["question","context"]
)

WEAVIATE_URL = os.environ['WEAVIATE_API_URL']
# WEAVIATE_API_KEY = os.environ['WEAVIATE_API_KEY']
# auth_config = weaviate.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY")
db = Weaviate.from_documents(docs, embeddings, weaviate_url=WEAVIATE_URL, by_text=False)

print(type(db),db)

# auth_config = weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
# client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=auth_config)

# vectorstore = Weaviate(client,"PodClip", "Content")

# print(type(vectorstore), vectorstore)

MyOpenAI = OpenAI(temperature=0.6)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# qa = ChatVectorDBChain.from_llm(MyOpenAI,db)

# chatgpt_chain = LLMChain(
#     llm=OpenAI(temperature=0.5),
#     prompt=_template
# )

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), db.as_retriever(), memory=memory, verbose=True, combine_docs_chain_kwargs={'prompt': _template})


while True:
    query = input("human :")
    result = qa({"question": query})
    print("AI :",result["answer"])











x = 10

def fun():
    # x = 5
    print(x)