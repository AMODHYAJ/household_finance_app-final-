import os
import logging
from typing import Optional, Dict, Any
import re

class AIProcessor:
    """
    Unified AI processor with fallback chain:
    1. Gemini (Primary)
    2. Groq (Secondary) 
    3. Rule-based (Final fallback)
    """
    
    def __init__(self):
        self.gemini_processor = None
        self.groq_processor = None
        self.available = False
        
        # Initialize processors in priority order
        self._initialize_processors()
        
    def _initialize_processors(self):
        """Initialize AI processors in priority order"""
        # Try Gemini first
        try:
            from agents.gemini_processor import GeminiProcessor
            self.gemini_processor = GeminiProcessor()
            if self.gemini_processor.available:
                print("✅ Gemini AI enabled (Primary)")
                self.available = True
                return
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")
        
        # Try Groq as fallback
        try:
            from agents.groq_processor import GroqProcessor
            self.groq_processor = GroqProcessor()
            if self.groq_processor.available:
                print("✅ Groq AI enabled (Fallback)")
                self.available = True
                return
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")
        
        print("⚠️ No AI processors available, using rule-based only")
    
    def categorize_transaction(self, note: str, amount: float) -> str:
        """Categorize transaction using AI with fallbacks"""
        if not note:
            return self._rule_based_categorization(note, amount)
        
        # Try Gemini first
        if self.gemini_processor and self.gemini_processor.available:
            try:
                category = self.gemini_processor.categorize_transaction(note, amount)
                if category and category != "other":
                    print(f"🤖 Gemini categorized: '{note}' → {category}")
                    return category
            except Exception as e:
                print(f"❌ Gemini categorization failed: {e}")
        
        # Try Groq as fallback
        if self.groq_processor and self.groq_processor.available:
            try:
                category = self.groq_processor.categorize_transaction(note, amount)
                if category and category != "other":
                    print(f"🤖 Groq categorized: '{note}' → {category}")
                    return category
            except Exception as e:
                print(f"❌ Groq categorization failed: {e}")
        
        # Final fallback to rule-based
        category = self._rule_based_categorization(note, amount)
        print(f"📝 Rule-based categorized: '{note}' → {category}")
        return category
    
    def generate_financial_insights(self, transactions_data: list) -> str:
        """Generate financial insights using AI with fallbacks"""
        # Try Gemini first
        if self.gemini_processor and self.gemini_processor.available:
            try:
                insights = self.gemini_processor.generate_financial_insights(transactions_data)
                if insights and "unavailable" not in insights.lower():
                    return insights
            except Exception as e:
                print(f"❌ Gemini insights failed: {e}")
        
        # Try Groq as fallback
        if self.groq_processor and self.groq_processor.available:
            try:
                insights = self.groq_processor.generate_financial_insights(transactions_data)
                if insights:
                    return insights
            except Exception as e:
                print(f"❌ Groq insights failed: {e}")
        
        # Final fallback to basic insights
        return self._generate_basic_insights(transactions_data)
    
    def _rule_based_categorization(self, note: str, amount: float) -> str:
        """Rule-based categorization fallback"""
        if not note:
            return "other"
        
        note_lower = note.lower()
        categories = {
            'food': ['starbucks', 'restaurant', 'mcdonald', 'coffee', 'meal', 'pizza', 'burger', 'cafe', 'grocer', 'food', 'dining', 'lunch', 'dinner'],
            'transport': ['uber', 'bus', 'taxi', 'train', 'flight', 'cab', 'lyft', 'gas', 'fuel', 'transport', 'commute'],
            'shopping': ['amazon', 'flipkart', 'clothes', 'shoes', 'book', 'gift', 'walmart', 'target', 'shopping', 'purchase'],
            'entertainment': ['movie', 'netflix', 'spotify', 'game', 'concert', 'cinema', 'entertainment'],
            'bills': ['electricity', 'water', 'internet', 'phone', 'rent', 'mortgage', 'bill', 'utility'],
            'healthcare': ['doctor', 'hospital', 'pharmacy', 'medical', 'insurance', 'health'],
            'income': ['salary', 'bonus', 'freelance', 'payment', 'invoice', 'stipend', 'refund', 'income']
        }
        
        for category, keywords in categories.items():
            if any(keyword in note_lower for keyword in keywords):
                return category
        
        return "other"
    
    def _generate_basic_insights(self, transactions_data: list) -> str:
        """Generate basic insights when AI is unavailable"""
        if not transactions_data:
            return "Add some transactions to get financial insights."
        
        import pandas as pd
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
        
        # Add category insights
        if not df[df['type'] == 'expense'].empty:
            top_category = df[df['type'] == 'expense'].groupby('category')['amount'].sum().idxmax()
            insights.append(f"📊 Your highest spending is in {top_category}.")
        
        insights.append(f"💼 You have {len(transactions_data)} transactions tracked.")
        
        return "\n".join(insights)
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get status of all AI processors"""
        status = {
            "gemini_available": bool(self.gemini_processor and self.gemini_processor.available),
            "groq_available": bool(self.groq_processor and self.groq_processor.available),
            "any_ai_available": self.available,
            "primary_ai": "Gemini" if self.gemini_processor and self.gemini_processor.available else 
                         "Groq" if self.groq_processor and self.groq_processor.available else 
                         "Rule-based"
        }
        
        if self.gemini_processor and self.gemini_processor.available:
            status["gemini_model"] = self.gemini_processor.model_name
        if self.groq_processor and self.groq_processor.available:
            status["groq_model"] = self.groq_processor.model_name
            
        return status