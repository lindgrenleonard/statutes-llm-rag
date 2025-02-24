from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader('documents').load_data()

index = VectorStoreIndex.from_documents(documents)

llm = Ollama(model="llama3.1:8b", request_timeout=60.0)

chat_engine = index.as_chat_engine(
    verbose=True,
    llm=llm,
)

chat_engine.chat_repl()