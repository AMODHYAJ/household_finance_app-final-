from cryptography.fernet import Fernet
import secrets
import string

def generate_keys():
    """Generate secure Fernet key and secret key"""
    
    # Generate Fernet key
    fernet_key = Fernet.generate_key().decode()
    print(f"🔐 FERNET_KEY: {fernet_key}")
    
    # Generate secret key (64 characters)
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()"
    secret_key = ''.join(secrets.choice(alphabet) for _ in range(64))
    print(f"🔑 SECRET_KEY: {secret_key}")
    
    # Write to .env file
    env_content = f"""# Security Keys - Generated Automatically
SECRET_KEY={secret_key}
FERNET_KEY={fernet_key}

# Gemini API (Optional - comment out if not using)
GEMINI_API_KEY=your-gemini-api-key-here
ENABLE_LLM_FEATURES=False

# Database
DATABASE_URL=sqlite:///finance.db
USE_REAL_DATA=True
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("\n✅ Keys generated and saved to .env file!")
    print("📁 Make sure to keep your .env file secure and never commit it to version control!")

if __name__ == "__main__":
    generate_keys()