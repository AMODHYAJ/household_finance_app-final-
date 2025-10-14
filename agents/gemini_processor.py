import google.generativeai as genai
import os
import logging
from typing import List, Dict, Any
import re
from utils.category_standardizer import category_standardizer

class GeminiProcessor:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = False
        self.model_name = None
        
        if self.api_key and not self.api_key.startswith('your-'):
            try:
                genai.configure(api_key=self.api_key)
                
                # List available models
                print("🔍 Checking available Gemini models...")
                models = genai.list_models()
                available_models = [model.name for model in models]
                print(f"📋 Found {len(available_models)} models")
                
                # Try to find a working text generation model
                preferred_models = [
                    'models/gemini-2.0-flash',
                    'models/gemini-2.0-flash-001',
                    'models/gemini-2.5-flash',
                    'models/gemini-2.5-flash-preview-09-2025',
                    'models/gemini-pro',
                    'models/gemini-1.5-flash'
                ]
                
                for model_name in preferred_models:
                    if model_name in available_models:
                        self.model_name = model_name
                        self.available = True
                        print(f"✅ Using Gemini model: {self.model_name}")
                        break
                
                if not self.available and available_models:
                    # Use any available model
                    for model_name in available_models:
                        if 'gemini' in model_name.lower():
                            self.model_name = model_name
                            self.available = True
                            print(f"✅ Using available Gemini model: {self.model_name}")
                            break
                
                if not self.available:
                    print("❌ No suitable Gemini model found")
                    
            except Exception as e:
                print(f"❌ Gemini API error: {e}")
                self.available = False
        else:
            print("⚠️ Gemini API key not configured")
            self.available = False
    
    def categorize_transaction(self, note: str, amount: float) -> str:
        """Categorize transaction using Gemini with standardization"""
        if not self.available:
            return "Other"
    
        try:
            model = genai.GenerativeModel(self.model_name)
        
            prompt = f"""
            Categorize this financial transaction into ONE category:
        
            Transaction: "{note}"
            Amount: ${amount}
        
            Categories: {', '.join(category_standardizer.get_standard_categories())}
        
            Return ONLY the category name exactly as shown above.
            Example: "Food"
            """
        
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=20,
                    temperature=0.1
                )
            )
        
            raw_category = response.text.strip().replace('"', '').replace("'", "")
            standardized = category_standardizer.standardize(raw_category)
        
            print(f"🤖 Gemini raw: '{raw_category}' → standardized: '{standardized}'")
            return standardized
        
        except Exception as e:
            logging.error(f"Gemini categorization failed: {e}")
            return "Other"
    
    def generate_financial_insights(self, transactions_data: List[Dict]) -> str:
        """Generate financial insights using Gemini"""
        if not self.available or not transactions_data:
            return "AI insights unavailable. Add more transactions to get insights."
        
        try:
            # Prepare summary for Gemini
            summary = self.prepare_transaction_summary(transactions_data)
            
            model = genai.GenerativeModel(self.model_name)
            
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
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=600,
                    temperature=0.7
                )
            )
            
            insights = response.text.strip()
            print(f"🤖 Gemini generated insights: {len(insights)} characters")
            return insights
            
        except Exception as e:
            logging.error(f"Gemini insights generation failed: {e}")
            return "Unable to generate AI insights at this time."
    
    def prepare_transaction_summary(self, transactions_data: List[Dict]) -> str:
        """Prepare transaction summary for Gemini analysis"""
        import pandas as pd
        
        if not transactions_data:
            return "No transaction data available."
            
        df = pd.DataFrame(transactions_data)
        
        total_income = df[df['type'] == 'income']['amount'].sum()
        total_expenses = df[df['type'] == 'expense']['amount'].sum()
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        # Category breakdown
        expense_by_category = df[df['type'] == 'expense'].groupby('category')['amount'].sum()
        
        # Top categories
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
        
        # Add time period if dates are available
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
                summary += f"- Analysis Period: {date_range}\n"
            except:
                pass
        
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
            'eating out': 'food',
            'rent': 'bills',
            'mortgage': 'bills',
            'accommodation': 'bills',
            'transportation': 'transport',
            'travel': 'transport',
            'commute': 'transport',
            'medical': 'healthcare',
            'health': 'healthcare',
            'school': 'education',
            'tuition': 'education',
            'fun': 'entertainment',
            'leisure': 'entertainment',
            'recreation': 'entertainment',
            'purchase': 'shopping',
            'retail': 'shopping',
            'utilities': 'bills',
            'utility': 'bills',
            'subscription': 'bills',
            'income': 'salary',
            'paycheck': 'salary',
            'wage': 'salary',
            'earning': 'salary'
        }
        
        for key, value in category_mapping.items():
            if key in category.lower():
                return value
        
        return 'other'