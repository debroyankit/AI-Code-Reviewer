import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment Variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
REPO_NAME = os.getenv("GITHUB_REPOSITORY")  # owner/repo
PR_NUMBER = int(os.getenv("PR_NUMBER", "0"))

def check_env_vars():
    if not (GITHUB_TOKEN and GROQ_API_KEY and REPO_NAME and PR_NUMBER):
        raise SystemExit("Missing required env vars: GITHUB_TOKEN, GROQ_API_KEY, GITHUB_REPOSITORY, PR_NUMBER")

# Configuration Constants
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".scala",
    ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".swift",
    ".php", ".rb", ".lua", ".pl", ".sh", ".bat",
    ".html", ".css", ".sql", ".json", ".yaml", ".yml", ".toml",
}

IGNORE_DIRS = {"node_modules", "venv", "env", ".git", "__pycache__", "dist", "build", "target"}