from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Postgres
    database_url: str = "postgresql+psycopg://docuagent:docuagent@localhost:5432/docuagent"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docuagent_chunks"

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"

    # Embeddings
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Reranker
    rerank_model: str = "BAAI/bge-reranker-base"

    # Self-healing loop
    grade_threshold: float = 0.7
    max_attempts: int = 3

    # Retrieval
    retrieval_top_k: int = 5

    # Langfuse
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


settings = Settings()
