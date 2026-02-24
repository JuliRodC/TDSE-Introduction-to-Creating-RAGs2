import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec

os.environ["GROQ_API_KEY"]     = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_2orQyX_F9DsJFSFuAkMGAxQe6Aga9Hf9smh6eEmxYC72waTBcZFDXPqxYKvCGjMC8fWpB3"

INDEX_NAME = "rag-index"

with open("documento.txt", "w", encoding="utf-8") as f:
    f.write("""
La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas.
Estos procesos incluyen el aprendizaje, el razonamiento y la autocorrección.

El machine learning es una rama de la IA que permite a las máquinas aprender de los datos sin ser programadas explícitamente.
Los algoritmos de machine learning identifican patrones en los datos y toman decisiones con mínima intervención humana.

El deep learning es un subconjunto del machine learning que utiliza redes neuronales artificiales con múltiples capas.
Estas redes pueden aprender representaciones complejas de los datos, lo que las hace muy poderosas para tareas como reconocimiento de imágenes y procesamiento de lenguaje natural.

Los modelos de lenguaje grande (LLMs) son sistemas de IA entrenados con enormes cantidades de texto.
Ejemplos de LLMs incluyen GPT-4 de OpenAI, Gemini de Google y Claude de Anthropic.
Estos modelos pueden generar texto, responder preguntas, traducir idiomas y realizar muchas otras tareas.

RAG (Retrieval-Augmented Generation) es una técnica que combina la búsqueda de información con la generación de texto.
En un sistema RAG, primero se recuperan documentos relevantes de una base de datos y luego se usan como contexto para que el LLM genere una respuesta más precisa y actualizada.

Pinecone es una base de datos vectorial que permite almacenar y buscar embeddings de forma eficiente.
Los embeddings son representaciones numéricas del texto que capturan su significado semántico.
""")

print("Cargando documento...")
loader = TextLoader("documento.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Documento dividido en {len(chunks)} fragmentos")

print("Creando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Guardando en Pinecone...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)
print("Documentos guardados en Pinecone!")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.5,
    max_tokens=1000
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
Responde la pregunta basándote únicamente en el siguiente contexto:

{context}

Pregunta: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("\n" + "="*50)
print("Sistema RAG listo. Haciendo preguntas...\n")

preguntas = [
    "¿Qué es el machine learning?",
    "¿Qué es RAG y para qué sirve?",
    "¿Qué es Pinecone?"
]

for pregunta in preguntas:
    print(f"Pregunta: {pregunta}")
    respuesta = rag_chain.invoke(pregunta)
    print(f"Respuesta: {respuesta}")
    print("-"*50)