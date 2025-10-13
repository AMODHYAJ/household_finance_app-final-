import openai
import os
from flask import current_app

class LLMAgent:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY', '')
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.available = True
        else:
            self.available = False
            print("⚠️ OpenAI API key not found. LLM features disabled.")
    
    def categorize_transaction_advanced(self, transaction_note, amount, date):
        """Advanced category prediction using LLM"""
        if not self.available:
            return "other"
            
        prompt = f"""
        Categorize this financial transaction:
        Note: "{transaction_note}"
        Amount: ${amount}
        Date: {date}
        
        Choose from: food, shopping, bills, entertainment, transport, healthcare, education, income, other
        
        Respond with ONLY the category name.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial categorization expert. Respond with only the category name."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.3
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"LLM categorization failed: {e}")
            return "other"