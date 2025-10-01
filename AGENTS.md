# AGENTS.md - Health Insurance Backend Project

## Commands
- **Development server**: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` or `bash run.sh`
- **Docker**: `docker-compose up` (includes Qdrant vector database)
- **Install dependencies**: `pip install -r requirements_new.txt`
- **Ingestion script**: `bash run_ingestion.sh`

## Architecture
- **FastAPI** backend with insurance chatbot functionality
- **Vector database**: Qdrant for document similarity search and RAG
- **LLM integration**: OpenAI GPT with Helicone monitoring
- **Storage**: AWS S3 for document storage
- **Core modules**: `app/` (main), `ingestion/` (data processing), `Insurance/` (domain logic)
- **Key services**: insurance_service, session_service, gemini_services, llm_services
- **Database**: Session management with SQLAlchemy, Qdrant for vector storage

## Code Style
- **Models**: Pydantic schemas with Field() validation, descriptive docstrings
- **Imports**: Standard library first, third-party, then local imports
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Documentation**: Triple-quoted docstrings for all classes/functions
- **Error handling**: Custom exception handlers in main.py, HTTPException for API errors
- **Configuration**: Centralized in config.py using pydantic-settings with .env support
- **Type hints**: Required for all function parameters and returns
