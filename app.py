from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader('documents').load_data()

index = VectorStoreIndex.from_documents(documents)

llm = Ollama(model="statutes-assitant:latest", request_timeout=60.0)

chat_engine = index.as_chat_engine(
    verbose=True,
    llm=llm,
    similarity_top_k=3,
    response_mode="compact",
    streaming=False,  
)

chat_engine.chat_repl()