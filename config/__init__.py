import os

class Settings:
    # NLP & LLM Settings
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    
    # Security Settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-super-secure-key-here'
    FERNET_KEY = os.environ.get('FERNET_KEY') or 'your-fernet-key-here'  # Simple string fallback
    
    # Dataset Settings
    REAL_DATASET_PATH = os.environ.get('REAL_DATASET_PATH', 'data/raw/household_income_dataset.csv')
    USE_REAL_DATA = os.environ.get('USE_REAL_DATA', 'True').lower() == 'true'

settings = Settings()