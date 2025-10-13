import pandas as pd
from datetime import datetime, timedelta
from core.database import Transaction
from flask_login import current_user

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

    # ------------------ Basic Methods ------------------
    def get_transactions_df(self, user_id):
        """Get transactions as DataFrame"""
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        if not transactions:
            return pd.DataFrame()
        
        data = [{
            "id": t.id,
            "date": t.date,
            "type": t.t_type,
            "category": t.category,
            "amount": t.amount,
            "note": t.note
        } for t in transactions]
        
        df = pd.DataFrame(data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df

    def get_user_transactions(self, user_id):
        """Get all transactions for user"""
        return Transaction.query.filter_by(user_id=user_id).all()

    # ------------------ Enhanced Categorization ------------------
    def suggest_category(self, note):
        """Smart categorization with AI fallback chain"""
        if not note:
            return "Other"
        
        if self.use_llm and self.ai_processor:
            # Use AI processor with fallback chain
            category = self.ai_processor.categorize_transaction(note, 0)
            return category.title()  # Convert to title case for display
        else:
            # Fallback to rule-based
            return self.auto_categorize(note)

    def auto_categorize(self, note):
        """Rule-based categorization (final fallback)"""
        if not note:
            return "Other"
        
        note_lower = note.lower()
        categories = {
            'Food': ['starbucks', 'restaurant', 'mcdonald', 'coffee', 'meal', 'pizza', 'burger', 'cafe', 'grocer'],
            'Transport': ['uber', 'bus', 'taxi', 'train', 'flight', 'cab', 'lyft', 'gas', 'fuel'],
            'Income': ['salary', 'bonus', 'freelance', 'payment', 'invoice', 'stipend', 'refund'],
            'Shopping': ['amazon', 'flipkart', 'clothes', 'shoes', 'book', 'gift', 'walmart', 'target'],
            'Entertainment': ['movie', 'netflix', 'spotify', 'game', 'concert', 'cinema'],
            'Bills': ['electricity', 'water', 'internet', 'phone', 'rent', 'mortgage'],
            'Healthcare': ['doctor', 'hospital', 'pharmacy', 'medical', 'insurance']
        }
        
        for category, keywords in categories.items():
            if any(keyword in note_lower for keyword in keywords):
                return category
        
        return "Other"

    def suggest_transaction_type(self, note):
        """Suggest transaction type based on note"""
        if not note:
            return "Expense"
            
        note_lower = note.lower()
        income_keywords = ['salary', 'bonus', 'freelance', 'payment', 'invoice', 'stipend', 'refund', 'received']
        
        if any(k in note_lower for k in income_keywords):
            return "Income"
        return "Expense"

    # ------------------ Validation & Analysis ------------------
    def validate_transaction(self, amount, t_type):
        """Validate transaction amount"""
        if amount <= 0:
            raise ValueError(f"{t_type} amount must be positive")
        if amount > 1_000_000:
            print("Warning: unusually high amount")
        return True

    def detect_anomalies(self, user_id):
        """Detect anomalous transactions using statistical methods"""
        df = self.get_transactions_df(user_id)
        if df.empty:
            return []
        
        # Calculate z-scores for anomaly detection
        mean = df['amount'].mean()
        std = df['amount'].std()
        
        if std > 0:  # Avoid division by zero
            anomalies = df[df['amount'] > mean + 2 * std]  # > 2 standard deviations
            return anomalies.to_dict('records')
        return []

    def explain_anomaly(self, transaction):
        """Explain why a transaction is anomalous"""
        note = transaction.note or ""
        amount = transaction.amount
        
        # Rule-based explanations
        if amount > 1000:
            return "⚠️ Unusually high transaction amount"
        elif "refund" in note.lower():
            return "🔄 This appears to be a refund"
        elif any(word in note.lower() for word in ["emergency", "urgent", "medical"]):
            return "🏥 Possible emergency expense"
        else:
            return "📊 Statistically unusual transaction amount"

    # ------------------ Summary & Reporting ------------------
    def generate_summary(self, user_id, period='week'):
        """Generate financial summary for given period"""
        df = self.get_transactions_df(user_id)
        if df.empty:
            return {}
        
        # Filter by period
        today = datetime.today().date()
        if period == 'week':
            cutoff = today - timedelta(days=7)
        elif period == 'month':
            cutoff = today - timedelta(days=30)
        else:
            cutoff = datetime.min.date()
        
        # Ensure cutoff is comparable with date column
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

    def suggest_notes(self, user_id, t_type):
        """Get unique note suggestions for autocomplete"""
        df = self.get_transactions_df(user_id)
        if df.empty:
            return []
        
        filtered_df = df[df['type'].str.lower() == t_type.lower()]
        return filtered_df['note'].dropna().unique().tolist()
    
    def get_ai_status(self):
        """Get AI processor status"""
        if self.ai_processor:
            return self.ai_processor.get_ai_status()
        return {"any_ai_available": False, "primary_ai": "Rule-based"}