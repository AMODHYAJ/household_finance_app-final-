import spacy
import pandas as pd
import re
from typing import List, Dict, Tuple, Any
import os
import logging
from datetime import datetime
from textblob import TextBlob

class NLPProcessor:
    def __init__(self):
        # Load spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️ spaCy model not found. Using fallback NLP")
            self.nlp = None
        
        # Initialize Gemini
        self.gemini_processor = None
        self.llm_available = False
        
        try:
            from agents.gemini_processor import GeminiProcessor
            self.gemini_processor = GeminiProcessor()
            if self.gemini_processor.available:
                self.llm_available = True
                print("✅ Gemini AI enabled")
            else:
                print("⚠️ Gemini AI unavailable")
        except Exception as e:
            print(f"⚠️ Could not load Gemini: {e}")
        
        # Enhanced rule-based patterns
        self.category_patterns = {
            'food': [r'\b(food|grocery|restaurant|dining|meal|supermarket|cafe|coffee|bakery|pizza|burger)\b'],
            'shopping': [r'\b(shopping|store|mall|retail|purchase|buy|amazon|ebay)\b'],
            'bills': [r'\b(bill|utility|electric|water|internet|phone|mobile|wifi)\b'],
            'entertainment': [r'\b(entertainment|movie|game|netflix|spotify|fun|hobby|clash of clans|gaming)\b'],
            'transport': [r'\b(transport|fuel|gas|bus|train|taxi|uber|lyft|petrol|car)\b'],
            'healthcare': [r'\b(health|medical|doctor|hospital|pharmacy|insurance|clinic)\b'],
            'education': [r'\b(education|school|university|course|book|tuition|college)\b'],
            'salary': [r'\b(salary|income|paycheck|payment|wage|earnings)\b']
        }
    
    def categorize_transaction_llm(self, note: str, amount: float) -> str:
        """Use Gemini AI to categorize transactions"""
        if self.llm_available and self.gemini_processor:
            try:
                return self.gemini_processor.categorize_transaction(note, amount)
            except Exception as e:
                logging.error(f"Gemini categorization failed: {e}")
        
        # Fallback to rule-based
        return self.categorize_transaction_basic(note, amount)
    
    def categorize_transaction_basic(self, note: str, amount: float) -> str:
        """Basic rule-based categorization"""
        note_lower = note.lower()
        
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if re.search(pattern, note_lower, re.IGNORECASE):
                    return category
        
        return 'other'
    
    def generate_financial_insights_llm(self, transactions_data: List[Dict]) -> str:
        """Generate financial insights using Gemini"""
        if self.llm_available and self.gemini_processor:
            try:
                return self.gemini_processor.generate_financial_insights(transactions_data)
            except Exception as e:
                logging.error(f"Gemini insights failed: {e}")
        
        # Fallback to rule-based insights
        return self.generate_basic_insights(transactions_data)
    
    def generate_basic_insights(self, transactions_data: List[Dict]) -> str:
        """Generate basic insights without AI"""
        if not transactions_data:
            return "Add some transactions to get financial insights."
        
        df = pd.DataFrame(transactions_data)
        
        total_income = df[df['type'] == 'income']['amount'].sum()
        total_expenses = df[df['type'] == 'expense']['amount'].sum()
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        insights = []
        
        if savings_rate > 20:
            insights.append("🎉 Excellent! You're saving more than 20% of your income.")
        elif savings_rate > 10:
            insights.append("👍 Good savings rate. Consider increasing to 20% for better financial security.")
        elif savings_rate > 0:
            insights.append("💡 You're saving money. Aim for 10-20% savings rate.")
        else:
            insights.append("⚠️ You're spending more than you earn. Review your expenses.")
        
        if len(transactions_data) > 5:
            insights.append(f"📊 You have {len(transactions_data)} transactions tracked.")
        
        return "\n".join(insights)
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract financial entities using NLP"""
        if not text or not self.nlp:
            return {}
        
        doc = self.nlp(text)
        entities = {
            'amounts': [],
            'currencies': [],
            'dates': [],
            'organizations': [],
            'products': []
        }
        
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                entities['amounts'].append(ent.text)
            elif ent.label_ == "DATE":
                entities['dates'].append(ent.text)
            elif ent.label_ == "ORG":
                entities['organizations'].append(ent.text)
            elif ent.label_ == "PRODUCT":
                entities['products'].append(ent.text)
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not text:
            return {'sentiment': 'neutral', 'polarity': 0, 'subjectivity': 0}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3)
            }
        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            return {'sentiment': 'neutral', 'polarity': 0, 'subjectivity': 0}
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Alias for extract_financial_entities"""
        return self.extract_financial_entities(text)