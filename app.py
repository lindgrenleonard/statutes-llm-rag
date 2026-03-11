import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.ollama import OllamaEmbedding

PERSIST_DIR = "./storage"

embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")
Settings.embed_model = embed_model

if os.path.exists(PERSIST_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
else:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    documents = SimpleDirectoryReader('documents').load_data()
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=3,
    device="cpu",
)

llm = Ollama(model="gemma3:4b-it-qat", request_timeout=300.0, num_ctx=4096, temperature=0.3)

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

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = body.get("message", "")

    def stream():
        response = chat_engine.stream_chat(message)
        for token in response.response_gen:
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        sources = []
        seen = set()
        for node in response.source_nodes:
            filename = node.metadata.get("file_name", "unknown")
            if filename not in seen:
                seen.add(filename)
                sources.append({
                    "file": filename,
                    "score": round(float(node.score), 3) if node.score is not None else None,
                })
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset():
    global chat_engine
    chat_engine = index.as_chat_engine(
        llm=llm,
        similarity_top_k=10,
        node_postprocessors=[reranker],
        response_mode="compact",
        streaming=True,
        system_prompt=SYSTEM_PROMPT,
    )
    return {"status": "ok"}
