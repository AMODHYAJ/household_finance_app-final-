from flask import Blueprint, render_template, request, jsonify, flash, current_app
from flask_login import login_required, current_user
from app import db
from core.database import Transaction
from datetime import datetime
import pandas as pd
import re

budget_bp = Blueprint('budget', __name__)

class BudgetManager:
    def __init__(self):
        self.default_budgets = {
            'food': 300,
            'shopping': 200,
            'bills': 150,
            'entertainment': 100,
            'transport': 100
        }
        
        # Enhanced category mapping
        self.category_keywords = {
            'food': ['food', 'grocery', 'restaurant', 'dining', 'meal', 'supermarket', 'cafe', 'pizza', 'burger'],
            'shopping': ['shopping', 'store', 'mall', 'retail', 'purchase', 'buy', 'amazon', 'ebay'],
            'bills': ['bills', 'utility', 'electric', 'water', 'internet', 'phone', 'rent', 'subscription', 'netflix'],
            'entertainment': ['entertainment', 'movie', 'game', 'netflix', 'spotify', 'fun', 'hobby', 'clash of clans', 'gaming'],
            'transport': ['transport', 'fuel', 'gas', 'bus', 'train', 'taxi', 'uber', 'car', 'petrol']
        }
    
    def match_category(self, transaction_category):
        """Match transaction category to budget category using keywords"""
        if not transaction_category:
            return 'other'
            
        transaction_category = str(transaction_category).lower().strip()
        
        for budget_category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in transaction_category:
                    return budget_category
        
        return 'other'
    
    def calculate_budget_progress(self, user_id, month=None, year=None):
        """Calculate budget usage for each category"""
        try:
            transactions = Transaction.query.filter_by(user_id=user_id).all()
            
            if not transactions:
                print("DEBUG: No transactions found for user")
                return {}
            
            # Use current month/year if not specified
            if month is None:
                month = datetime.now().month
            if year is None:
                year = datetime.now().year
            
            # Convert to DataFrame with proper date conversion
            data = []
            for t in transactions:
                if t.t_type == 'expense':  # Only track expenses for budgets
                    matched_category = self.match_category(t.category)
                    data.append({
                        'category': matched_category,
                        'amount': float(t.amount),
                        'date': t.date,
                        'original_category': t.category
                    })
                    print(f"DEBUG: Transaction '{t.category}' -> Budget category '{matched_category}', Date: {t.date}")
            
            if not data:
                print("DEBUG: No expense transactions found")
                return {}
                
            df = pd.DataFrame(data)
            print(f"DEBUG: Processing {len(df)} expense transactions")
            
            # Ensure date column is datetime
            if len(df) > 0 and not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Filter for specified month/year OR show all data if no matches
            filtered_expenses = df[
                (df['date'].dt.month == month) &
                (df['date'].dt.year == year)
            ]
            
            # If no expenses in current month, show all expenses as demonstration
            if filtered_expenses.empty:
                print(f"DEBUG: No expenses in {month}/{year}, showing all expenses")
                filtered_expenses = df
            
            if filtered_expenses.empty:
                print("DEBUG: No expenses found after filtering")
                return {}
            
            print(f"DEBUG: Found {len(filtered_expenses)} expenses to analyze")
            
            # Calculate spending by budget category
            category_spending = filtered_expenses.groupby('category')['amount'].sum().to_dict()
            print(f"DEBUG: Category spending: {category_spending}")
            
            # Calculate progress for all budget categories
            budget_progress = {}
            total_spent = 0
            
            for budget_category, budget_amount in self.default_budgets.items():
                spent = category_spending.get(budget_category, 0)
                total_spent += spent
                
                # Calculate progress
                progress = (spent / budget_amount) * 100 if budget_amount > 0 else 0
                remaining = max(0, budget_amount - spent)
                
                # Determine status
                if progress >= 100:
                    status = 'danger'
                elif progress >= 80:
                    status = 'warning'
                else:
                    status = 'success'
                
                budget_progress[budget_category] = {
                    'budget': budget_amount,
                    'spent': round(spent, 2),
                    'remaining': round(remaining, 2),
                    'progress': min(round(progress, 1), 100),
                    'status': status,
                    'transaction_count': len(filtered_expenses[filtered_expenses['category'] == budget_category])
                }
                
                print(f"DEBUG: {budget_category}: ${spent} of ${budget_amount} ({progress}%) - {budget_progress[budget_category]['transaction_count']} transactions")
            
            print(f"DEBUG: Total spent across all categories: ${total_spent}")
            return budget_progress
            
        except Exception as e:
            print(f"Budget calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
@budget_bp.route('/budget')
@login_required
def budget_page():
    try:
        budget_manager = BudgetManager()
        
        # Get month/year from query parameters or use current
        month = request.args.get('month', type=int)
        year = request.args.get('year', type=int)
        
        budget_progress = budget_manager.calculate_budget_progress(
            current_user.id, 
            month=month, 
            year=year
        )
        
        print(f"DEBUG: Budget progress result: {budget_progress}")
        
        # Get all user transactions for info
        all_transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        expense_transactions = [t for t in all_transactions if t.t_type == 'expense']
        
        return render_template('budget.html', 
                             title='Budget Management',
                             budget_progress=budget_progress,
                             default_budgets=budget_manager.default_budgets,
                             total_expenses=len(expense_transactions),
                             current_month=datetime.now().month,
                             current_year=datetime.now().year)
    except Exception as e:
        print(f"Budget page error: {e}")
        import traceback
        traceback.print_exc()
        return render_template('budget.html',
                             title='Budget Management',
                             budget_progress={},
                             default_budgets={},
                             total_expenses=0)

@budget_bp.route('/api/budget-data')
@login_required
def budget_data():
    try:
        budget_manager = BudgetManager()
        progress = budget_manager.calculate_budget_progress(current_user.id)
        return jsonify(progress)
    except Exception as e:
        return jsonify({'error': str(e)})

# Debug route to check transaction dates
@budget_bp.route('/budget-debug')
@login_required
def budget_debug():
    """Debug page to see transaction dates and types"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    transaction_data = []
    for t in transactions:
        transaction_data.append({
            'id': t.id,
            'type': t.t_type,
            'category': t.category,
            'amount': t.amount,
            'date': str(t.date),
            'date_type': str(type(t.date)),
            'date_month': t.date.month if hasattr(t.date, 'month') else 'N/A',
            'date_year': t.date.year if hasattr(t.date, 'year') else 'N/A'
        })
    
    return jsonify({
        'user_id': current_user.id,
        'total_transactions': len(transactions),
        'expense_transactions': len([t for t in transactions if t.t_type == 'expense']),
        'current_month': datetime.now().month,
        'current_year': datetime.now().year,
        'transactions': transaction_data
    })