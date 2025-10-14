import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
from core.database import Transaction
from flask_login import current_user
from utils.category_standardizer import category_standardizer

class DataCollectorAgent:
    def __init__(self, use_llm=True):
        self.current_user = None
        self.use_llm = use_llm
        self.ai_processor = None
        
        if use_llm:
            try:
                from agents.ai_processor import AIProcessor
                self.ai_processor = AIProcessor()
                print(f"✅ AI Processor initialized: {self.ai_processor.get_ai_status()['primary_ai']}")
            except Exception as e:
                print(f"❌ AI Processor initialization failed: {e}")
                self.use_llm = False

    def get_transactions_df(self, user_id):
        """Get transactions as DataFrame with consistent field names"""
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        if not transactions:
            return pd.DataFrame()
        
        data = [{
            "id": t.id,
            "date": t.date,
            "type": t.type,  # CHANGED: t_type → type
            "category": t.category,
            "amount": t.amount,
            "note": t.note
        } for t in transactions]
        
        df = pd.DataFrame(data)
        if not df.empty and 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def get_user_transactions(self, user_id):
        """Get all transactions for user"""
        return Transaction.query.filter_by(user_id=user_id).all()

    def suggest_category(self, note):
        """Smart categorization with standardization"""
        if not note:
            return "Other"
    
        if self.use_llm and self.ai_processor:
            try:
            # Get AI category
                ai_category = self.ai_processor.categorize_transaction(note, 0)
                # Standardize the AI result
                standardized = category_standardizer.standardize(ai_category)
                print(f"🤖 AI categorized: '{note}' → {ai_category} → {standardized}")
                return standardized
            except Exception as e:
                print(f"AI categorization failed: {e}")
    
        # Fallback to rule-based with standardization
        category = self.auto_categorize(note)
        standardized = category_standardizer.standardize(category)
        print(f"📝 Rule-based categorized: '{note}' → {category} → {standardized}")
        return standardized

    def auto_categorize(self, note):
        """Rule-based categorization with standardization"""
        if not note:
            return "Other"
    
        note_lower = note.lower()
    
        # Use the standardizer for consistency
        for standard_category, variations in category_standardizer.standard_categories.items():
            for variation in variations:
                if variation in note_lower:
                    return standard_category
    
        return "Other"

    def suggest_transaction_type(self, note):
        """Suggest transaction type based on note"""
        if not note:
            return "expense"
            
        note_lower = note.lower()
        income_keywords = ['salary', 'bonus', 'freelance', 'payment', 'invoice', 'stipend', 'refund', 'received']
        
        if any(k in note_lower for k in income_keywords):
            return "income"
        return "expense"

    def validate_transaction(self, amount, transaction_type):
        """Validate transaction amount"""
        if amount <= 0:
            raise ValueError(f"{transaction_type} amount must be positive")
        if amount > 1_000_000:
            print("Warning: unusually high amount")
        return True

    def detect_anomalies(self, user_id):
        """Detect anomalous transactions using statistical methods"""
        df = self.get_transactions_df(user_id)
        if df.empty:
            return []
        
        mean = df['amount'].mean()
        std = df['amount'].std()
        
        if std > 0:
            anomalies = df[df['amount'] > mean + 2 * std]
            return anomalies.to_dict('records')
        return []

    def explain_anomaly(self, transaction):
        """Explain why a transaction is anomalous"""
        note = transaction.note or ""
        amount = transaction.amount
        
        if amount > 1000:
            return "⚠️ Unusually high transaction amount"
        elif "refund" in note.lower():
            return "🔄 This appears to be a refund"
        elif any(word in note.lower() for word in ["emergency", "urgent", "medical"]):
            return "🏥 Possible emergency expense"
        else:
            return "📊 Statistically unusual transaction amount"

    def generate_summary(self, user_id, period='week'):
        """Generate financial summary for given period"""
        df = self.get_transactions_df(user_id)
        if df.empty:
            return {}
        
        today = datetime.today().date()
        if period == 'week':
            cutoff = today - timedelta(days=7)
        elif period == 'month':
            cutoff = today - timedelta(days=30)
        else:
            cutoff = datetime.min.date()
        
        if not df.empty:
            recent = df[df['date'].dt.date >= cutoff]
            
            total_income = recent[recent['type'].str.lower() == 'income']['amount'].sum()
            total_expense = recent[recent['type'].str.lower() == 'expense']['amount'].sum()
            balance = total_income - total_expense
            
            return {
                'total_income': total_income,
                'total_expense': total_expense,
                'balance': balance,
                'transaction_count': len(recent),
                'savings_rate': (balance / total_income * 100) if total_income > 0 else 0
            }
        return {}

    def suggest_notes(self, user_id, transaction_type):
        """Get unique note suggestions for autocomplete"""
        df = self.get_transactions_df(user_id)
        if df.empty:
            return []
        
        filtered_df = df[df['type'].str.lower() == transaction_type.lower()]
        return filtered_df['note'].dropna().unique().tolist()
    
    def get_ai_status(self):
        """Get AI processor status"""
        if self.ai_processor:
            return self.ai_processor.get_ai_status()
        return {"any_ai_available": False, "primary_ai": "Rule-based"}