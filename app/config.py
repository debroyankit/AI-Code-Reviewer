from typing import Set
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Read automatically from env or .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    github_token: str = Field(..., alias="GITHUB_TOKEN")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")  # Not needed for cache sync
    repo_name: str = Field(..., alias="GITHUB_REPOSITORY")
    pr_number: int = Field(0, alias="PR_NUMBER")  # 0 = sync mode (no PR)

    # Run mode: "review" (PR review) or "sync" (cache sync)
    run_mode: str = Field("review", alias="RUN_MODE")

    # Upgraded configs: Customizable thresholds
    max_diff_length: int = Field(30000, alias="MAX_DIFF_LENGTH")
    max_files_per_pr: int = Field(20, alias="MAX_FILES_PER_PR")
    max_lines_per_file: int = Field(500, alias="MAX_LINES_PER_FILE")
    llm_model: str = Field("llama-3.1-8b-instant", alias="LLM_MODEL")

    # Extension filters
    supported_extensions: Set[str] = {
        ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala",
        ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".swift",
        ".php", ".rb", ".lua", ".pl", ".sh", ".bat",
        ".html", ".css", ".sql", ".json", ".yaml", ".yml", ".toml",
    }
    
    ignore_dirs: Set[str] = {"node_modules", "venv", "env", ".git", "__pycache__", "dist", "build", "target", ".faiss_cache"}

    @field_validator("pr_number", mode="before")
    @classmethod
    def parse_pr_number(cls, v):
        if isinstance(v, str):
            if not v.strip():
                return 0
            return int(v)
        return v

# Instantiate a global settings object. It will throw an error immediately 
# if GITHUB_TOKEN or GITHUB_REPOSITORY are missing.
try:
    settings = Settings()
except Exception as e:
    import sys
    print(f"❌ Configuration error:\n{e}", file=sys.stderr)
    sys.exit(2)
