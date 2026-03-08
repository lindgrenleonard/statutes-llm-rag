import os
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

PERSIST_DIR = "./storage"

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", device="cpu")
Settings.embed_model = embed_model

if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model,
    )

    documents = SimpleDirectoryReader('documents').load_data()
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
    device="cpu",
)

llm = Ollama(model="gemma3:4b", request_timeout=300.0, num_ctx=4096, temperature=0.3)

SYSTEM_PROMPT = """You are a document assistant for the IT Chapter's statutes and memos.

Rules:
- Start with a direct answer, then cite the specific section or article.
- Use bullet points for multiple items.
- Keep responses concise but thorough. Say "For more details, refer to [Section X]" for lengthy topics.
- Only answer based on the provided context. If the context doesn't contain the answer, say so.
- Do not provide legal advice."""

chat_engine = index.as_chat_engine(
    llm=llm,
    similarity_top_k=10,
    node_postprocessors=[reranker],
    response_mode="compact",
    streaming=True,
    system_prompt=SYSTEM_PROMPT,
)

print("===== Entering Chat REPL =====")
print('Type "exit" to exit.\n')

while True:
    message = input("You: ")
    if message.strip().lower() == "exit":
        break

    response = chat_engine.stream_chat(message)
    print("\nAssistant: ", end="", flush=True)
    for token in response.response_gen:
        print(token, end="", flush=True)
    print()

    sources = response.source_nodes
    if sources:
        seen = set()
        print("\n📄 Sources:")
        for node in sources:
            filename = node.metadata.get("file_name", "unknown")
            if filename not in seen:
                seen.add(filename)
                score = f" (score: {node.score:.3f})" if node.score is not None else ""
                print(f"  - {filename}{score}")
    print()
