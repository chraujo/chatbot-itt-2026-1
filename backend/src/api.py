from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import asyncio
from contextlib import asynccontextmanager

from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .routers import chat_router, admin_router
from .routers.chat import limiter
from .dependencies import get_settings
from .services import VectorDB


async def startup_sync():
    """
    Função que roda em segundo plano quando o servidor liga.
    Ela recria a memória automaticamente.
    """
    print("Auto-Sync: Iniciando sincronização automática com Google Drive...")
    try:
        settings = get_settings()
        if settings.GOOGLE_DRIVE_FOLDER_ID and settings.GOOGLE_API_KEY:
            vector_db = VectorDB(settings)
            await asyncio.to_thread(vector_db.refresh_knowledge_base)
            print("Auto-Sync: Memória recriada com sucesso! O Chat está pronto.")
        else:
            print("Auto-Sync: Configurações do Drive incompletas. Pulando sincronização.")
    except Exception as e:
        print(f"Auto-Sync Falhou: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(startup_sync())
    yield
    print("Servidor desligando...")


settings = get_settings()

app = FastAPI(
    title="ITT Chatbot API",
    description="API for ITT institutional chatbot with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, settings.FRONTEND_DEV_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "ITT Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )