from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from datetime import datetime, timedelta
import jwt
from flask import request, current_app
import re

class AdvancedSecurity:
    def __init__(self):
        self.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key-change-in-production')
        
        # Handle Fernet key generation safely
        fernet_key = os.environ.get('FERNET_KEY')
        if not fernet_key:
            # Generate a proper Fernet key
            fernet_key = Fernet.generate_key().decode()
            print(f"🔐 Generated new Fernet key: {fernet_key}")
        else:
            # Ensure the key is properly formatted
            try:
                # Test if it's a valid Fernet key
                Fernet(fernet_key.encode())
                print("🔐 Using existing Fernet key")
            except (ValueError, Exception):
                # Generate a new key if invalid
                print("🔐 Invalid Fernet key, generating new one...")
                fernet_key = Fernet.generate_key().decode()
        
        # Ensure the key is bytes
        if isinstance(fernet_key, str):
            fernet_key = fernet_key.encode()
        
        self.fernet = Fernet(fernet_key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return ""
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            print(f"🔒 Encryption error: {e}")
            return data  # Fallback to plain text
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return ""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            print(f"🔒 Decryption error: {e}")
            return encrypted_data  # Fallback to return as-is
    
    # ... rest of your methods remain the same
    def generate_jwt_token(self, user_id: int, username: str, role: str) -> str:
        """Generate JWT token for authentication"""
        payload = {
            'user_id': user_id,
            'username': username,
            'role': role,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_jwt_token(self, token: str) -> dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            raise Exception("Token expired")
        except jwt.InvalidTokenError:
            raise Exception("Invalid token")
    
    def sanitize_input(self, input_string: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not input_string:
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[;\"\'\-\-]', '', input_string)
        # Limit length
        sanitized = sanitized[:500]
        
        return sanitized.strip()
    
    def validate_transaction_data(self, transaction_data: dict) -> tuple[bool, str]:
        """Validate transaction data for security"""
        try:
            # Check required fields
            required_fields = ['type', 'category', 'amount', 'date']
            for field in required_fields:
                if field not in transaction_data:
                    return False, f"Missing required field: {field}"
            
            # Validate transaction type
            if transaction_data['type'] not in ['income', 'expense']:
                return False, "Invalid transaction type"
            
            # Validate amount
            try:
                amount = float(transaction_data['amount'])
                if amount <= 0:
                    return False, "Amount must be positive"
                if amount > 1000000:  # Reasonable upper limit
                    return False, "Amount too large"
            except ValueError:
                return False, "Invalid amount format"
            
            # Sanitize category and note
            if 'category' in transaction_data:
                transaction_data['category'] = self.sanitize_input(transaction_data['category'])
            if 'note' in transaction_data:
                transaction_data['note'] = self.sanitize_input(transaction_data['note'])
            
            return True, "Validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def audit_log(self, user_id: int, action: str, details: str = ""):
        """Create audit log for security monitoring"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'action': action,
            'details': details,
            'ip_address': request.remote_addr if request else 'unknown'
        }
        
        # In production, this would write to a secure audit log
        print(f"🔒 AUDIT: {log_entry}")
        
        return log_entry