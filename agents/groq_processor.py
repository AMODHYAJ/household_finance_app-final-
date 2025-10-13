from groq import Groq
import os
import logging
from typing import List, Dict, Any
import re

class GroqProcessor:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        self.available = False
        self.client = None
        self.model_name = "llama-3.3-70b-versatile"
        
        if self.api_key and not self.api_key.startswith('your-'):
            try:
                self.client = Groq(api_key=self.api_key)
                self.available = True
                print("✅ Groq LLM enabled")
            except Exception as e:
                print(f"❌ Groq initialization failed: {e}")
                self.available = False
        else:
            print("⚠️ Groq API key not configured")
            self.available = False
    
    def categorize_transaction(self, note: str, amount: float) -> str:
        """Categorize transaction using Groq"""
        if not self.available:
            return "other"
        
        try:
            prompt = f"""
            Categorize this financial transaction into ONE category:
            
            Transaction: "{note}"
            Amount: ${amount}
            
            Categories: food, shopping, bills, entertainment, transport, healthcare, education, salary, other
            
            Return ONLY the category name in lowercase.
            Example: "food"
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial categorization expert. Respond with only the category name."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            category = response.choices[0].message.content.strip().lower()
            return self.validate_category(category)
            
        except Exception as e:
            logging.error(f"Groq categorization failed: {e}")
            return "other"
    
    def generate_financial_insights(self, transactions_data: List[Dict]) -> str:
        """Generate financial insights using Groq"""
        if not self.available or not transactions_data:
            return "AI insights unavailable. Add more transactions to get insights."
        
        try:
            summary = self.prepare_transaction_summary(transactions_data)
            
            prompt = f"""
            Analyze this household financial data and provide 3-5 key insights and recommendations:
            
            {summary}
            
            Please provide:
            1. Spending pattern analysis
            2. Savings opportunities  
            3. Financial health assessment
            4. Specific actionable recommendations
            
            Format as clear, concise bullet points. Be practical and helpful for household budgeting.
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial advisor providing practical insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.7
            )
            
            insights = response.choices[0].message.content.strip()
            print(f"🤖 Groq generated insights: {len(insights)} characters")
            return insights
            
        except Exception as e:
            logging.error(f"Groq insights generation failed: {e}")
            return "Unable to generate AI insights at this time."
    
    def prepare_transaction_summary(self, transactions_data: List[Dict]) -> str:
        """Prepare transaction summary for Groq analysis"""
        import pandas as pd
        
        if not transactions_data:
            return "No transaction data available."
            
        df = pd.DataFrame(transactions_data)
        
        total_income = df[df['type'] == 'income']['amount'].sum()
        total_expenses = df[df['type'] == 'expense']['amount'].sum()
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        expense_by_category = df[df['type'] == 'expense'].groupby('category')['amount'].sum()
        top_categories = expense_by_category.nlargest(3)
        
        summary = f"""
        HOUSEHOLD FINANCIAL ANALYSIS
        
        INCOME & EXPENSES:
        - Total Income: ${total_income:,.2f}
        - Total Expenses: ${total_expenses:,.2f}
        - Net Savings: ${net_savings:,.2f}
        - Savings Rate: {savings_rate:.1f}%
        
        TOP SPENDING CATEGORIES:
        """
        
        for category, amount in top_categories.items():
            percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
            summary += f"  - {category}: ${amount:,.2f} ({percentage:.1f}% of expenses)\n"
        
        summary += f"""
        TRANSACTION OVERVIEW:
        - Total Transactions: {len(transactions_data)}
        - Income Transactions: {len(df[df['type'] == 'income'])}
        - Expense Transactions: {len(df[df['type'] == 'expense'])}
        """
        
        return summary
    
    def validate_category(self, category: str) -> str:
        """Validate and standardize category"""
        valid_categories = ['food', 'shopping', 'bills', 'entertainment', 'transport', 'healthcare', 'education', 'salary', 'other']
        
        if category in valid_categories:
            return category
        
        # Map similar categories
        category_mapping = {
            'groceries': 'food',
            'grocery': 'food',
            'dining': 'food',
            'restaurant': 'food',
            'rent': 'bills',
            'mortgage': 'bills',
            'transportation': 'transport',
            'medical': 'healthcare',
            'income': 'salary',
            'paycheck': 'salary'
        }
        
        for key, value in category_mapping.items():
            if key in category.lower():
                return value
        
        return 'other'