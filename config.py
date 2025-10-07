import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the same directory
load_dotenv()

class Settings:
    """
    Application settings loaded from environment variables.
    """
    GEMINI_API_KEY: str

    def __init__(self):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please create a .env file with this key.")
        self.GEMINI_API_KEY = gemini_api_key
        self.EXA_API_KEY = os.getenv("EXA_API_KEY")
        if not self.EXA_API_KEY:
            print("⚠️  Warning: EXA_API_KEY not set. Web Bootstrap will be disabled.")

        # Database or other configs (optional)
        self.DATABASE_URL = os.getenv("DATABASE_URL")



# Create a single, importable instance of the settings
settings = Settings()