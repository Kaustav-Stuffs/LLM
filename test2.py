# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "llama-index",
#     "transformers",
#     "uvicorn",
# ]
# ///
from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.readers.file import SimpleDirectoryReader
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms.base import LLM

from llama_index.readers.file import SimpleDirectoryReader
from llama_index.llms.base import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import uvicorn

# === Load Qwen2.5:1.5B Locally ===
print("Loading Qwen2.5:1.5B model...")
model_id = "Qwen/Qwen1.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cpu")
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, max_new_tokens=300)
print("Model loaded successfully.")

# === LLM Wrapper ===
class QwenLLM(LLM):
    def complete(self, prompt: str, **kwargs) -> str:
        result = pipeline(prompt)[0]["generated_text"]
        return result[len(prompt):].strip()

# === Load & Index Documents ===
def build_index():
    print("Indexing documents from ./docs ...")
    documents = SimpleDirectoryReader("./docs").load_data()
    llm = QwenLLM()
    service_context = ServiceContext.from_defaults(llm=llm)
    return VectorStoreIndex.from_documents(documents, service_context=service_context)

index = build_index()
query_engine = index.as_query_engine()
print("Index ready.")

# === FastAPI App ===
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    query: str

@app.post("/customerSupport")
def customer_support(req: ChatRequest):
    try:
        response = query_engine.query(req.query)
        return {
            "user_id": req.user_id,
            "response": str(response)
        }
    except Exception as e:
        return {
            "user_id": req.user_id,
            "response": "Sorry, I couldn't process your question.",
            "error": str(e)
        }

# === Auto-run Uvicorn ===
if __name__ == "__main__":
    uvicorn.run("customer_support_api:app", host="0.0.0.0", port=5556, reload=False)
