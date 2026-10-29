from fastapi import FastAPI
app = FastAPI(
    title = "Hybrid Search RAG",
    version = "0.1.0",
    description = (
        "Evidence-grounded question answering"
        "over internal documents."
    ),
)
@app.get("/health")
def health_check() -> dict[str, str]:
    return{"status": "healthy"}