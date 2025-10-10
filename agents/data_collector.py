from core.database import Transaction
from flask_login import current_user

class DataCollectorAgent:
    def __init__(self):
        self.current_user = None

    def get_transactions_df(self, user_id):
        """Get transactions as DataFrame for a specific user"""
        transactions = Transaction.query.filter_by(user_id=user_id).all()
        
        if not transactions:
            return None
        
        data = [{
            "id": t.id,
            "date": t.date,
            "type": t.t_type,
            "category": t.category,
            "amount": t.amount,
            "note": t.note
        } for t in transactions]
        
        import pandas as pd
        return pd.DataFrame(data)

    def get_user_transactions(self, user_id):
        """Get transactions for a specific user"""
        return Transaction.query.filter_by(user_id=user_id).all()