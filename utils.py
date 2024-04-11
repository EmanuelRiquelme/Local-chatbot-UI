from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp

model_name = "hkunlp/instructor-xl"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
PROMPT_TEMPLATE = """
Answer the question based only in the following context and the chat history:
---
Chat history: {chat_history}
---
Context: {context}

---

Answer the question based on the above context: {question}
"""

n_gpu_layers = -1
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
model_path = 'mistral-7b-instruct-v0.2.Q4_K_M.gguf'
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    temperature=0.8,
    n_ctx=10000,
)
def create_vector_db(pdf_file):
    loader = PyPDFLoader(pdf_file)
    doc = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )
    chunks = text_splitter.split_documents(doc)
    DB_file_name = pdf_file.split('/')[-1].split('.')[0]
    db = Chroma.from_documents(
            chunks, hf, persist_directory=DB_file_name
        )
    db.persist()
        
def process_mesagges(chat_history):
    chat = ''
    print('chat_history')
    print(chat_history,type(chat_history))
    for message in chat_history:
        for role,text in message.items():
            chat +=(f'{role}: {text}\n')
    return chat

def get_prompt(db,prompt,chat_history,k = 9):
    results = db.similarity_search_with_relevance_scores(prompt, k=k)
    context_text= "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chat_history = process_mesagges(chat_history)
    prompt = prompt_template.format(chat_history = chat_history,context=context_text, question=prompt)
    return prompt

def start_db(DB_file_name,embedding_function = hf):
    return Chroma(persist_directory=DB_file_name, embedding_function=embedding_function)
def inference_pipeline(db,prompt,chat_history):
    prompt = get_prompt(db,prompt,chat_history)
    return llm.invoke(prompt)

