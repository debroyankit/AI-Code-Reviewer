import os
from dotenv import load_dotenv
from github import Github
from langchain_groq import ChatGroq

class Env:
    def __init__(self):
        self.GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.REPO_NAME = os.getenv("GITHUB_REPOSITORY")
        self.PR_NUMBER = int(os.getenv("PR_NUMBER"))
        
        self.gh = Github(self.GITHUB_TOKEN)
        self.repo = self.gh.get_repo(self.REPO_NAME)
        self.pr = self.repo.get_pull(self.PR_NUMBER)

        self.llm = ChatGroq(
            groq_api_key=self.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

def load_env():
    load_dotenv()
    return Env()
