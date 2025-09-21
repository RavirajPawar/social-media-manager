from fastapi import FastAPI
from app.core.config import settings
from app.api.endpoints import generate

app = FastAPI(
    title="Social Media Content Engine API",
    description="API for generating platform-specific social media content",
    version="1.0.0",
)

# Include routers
app.include_router(generate.router, prefix="/api/v1", tags=["Generation"])


@app.get("/")
async def root():
    return {
        "message": "Social Media Content Engine API is running",
        "model": settings.openai_model_name,
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
