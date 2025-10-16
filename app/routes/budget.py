from flask import Blueprint, Response, render_template, request, jsonify, flash, current_app
from flask_login import login_required, current_user
from app import db
from core.database import Transaction, UserBudget
from datetime import datetime
import pandas as pd
import re

import sys
import os
# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.category_standardizer import CategoryStandardizer, category_standardizer

budget_bp = Blueprint('budget', __name__)

class BudgetManager:
    def __init__(self):
        self.default_budgets = {
            'Food': 300,
            'Shopping': 200,
            'Bills': 150,
            'Entertainment': 100,
            'Transport': 100,
            'Other': 200  # Added Other category with a default budget
        }

    def get_user_budgets(self, user_id):
        """Get user-specific budgets from database - FIXED VERSION"""
        user_budgets = UserBudget.query.filter_by(user_id=user_id).first()
        
        if user_budgets and user_budgets.budgets:
            # Merge user budgets with default budgets to ensure all categories exist
            merged_budgets = self.default_budgets.copy()  # Start with defaults
            merged_budgets.update(user_budgets.budgets)   # Override with user values
            print(f"DEBUG: Using merged budgets: {merged_budgets}")
            return merged_budgets
        
        print(f"DEBUG: Using default budgets: {self.default_budgets}")
        return self.default_budgets

    def update_user_budget(self, user_id, category, amount):
        """Allow users to set custom budgets - FIXED VERSION"""
        user_budget = UserBudget.query.filter_by(user_id=user_id).first()
        if not user_budget:
            # Start with default budgets and update the specific category
            user_budget = UserBudget(user_id=user_id, budgets=self.default_budgets.copy())
        
        # Update the specific category
        user_budget.budgets[category] = amount
        user_budget.updated_at = datetime.utcnow()
        
        db.session.add(user_budget)
        db.session.commit()
        print(f"DEBUG: Updated budget for {category}: ${amount}")

    def match_category(self, transaction_category):
        """Match transaction category to budget category - ENHANCED"""
        if not transaction_category:
            return 'Other'
        
        transaction_lower = transaction_category.lower().strip()
        
        print(f"DEBUG: Matching category '{transaction_category}' -> '{transaction_lower}'")
        
        # Enhanced category mapping with more comprehensive matching
        category_map = {
            # Food category
            'food': 'Food',
            'groceries': 'Food',
            'restaurant': 'Food',
            'dining': 'Food',
            'meal': 'Food',
            'lunch': 'Food',
            'dinner': 'Food',
            'breakfast': 'Food',
            'cafe': 'Food',
            'coffee': 'Food',
            'grocery': 'Food',
            'supermarket': 'Food',
            
            # Shopping category
            'shopping': 'Shopping',
            'amazon': 'Shopping',
            'retail': 'Shopping',
            'purchase': 'Shopping',
            'store': 'Shopping',
            'mall': 'Shopping',
            'clothing': 'Shopping',
            'apparel': 'Shopping',
            'merchandise': 'Shopping',
            
            # Bills category
            'bills': 'Bills',
            'utilities': 'Bills',
            'electricity': 'Bills',
            'internet': 'Bills',
            'phone': 'Bills',
            'subscription': 'Bills',
            'rent': 'Bills',
            'mortgage': 'Bills',
            'insurance': 'Bills',
            'water': 'Bills',
            'gas bill': 'Bills',
            'electric': 'Bills',
            
            # Entertainment category
            'entertainment': 'Entertainment',
            'movie': 'Entertainment',
            'netflix': 'Entertainment',
            'gaming': 'Entertainment',
            'game': 'Entertainment',
            'spotify': 'Entertainment',
            'music': 'Entertainment',
            'concert': 'Entertainment',
            'hobby': 'Entertainment',
            'clash of clans': 'Entertainment',
            'streaming': 'Entertainment',
            'theater': 'Entertainment',
            
            # Transport category
            'transport': 'Transport',
            'uber': 'Transport',
            'taxi': 'Transport',
            'bus': 'Transport',
            'train': 'Transport',
            'gas': 'Transport',
            'fuel': 'Transport',
            'petrol': 'Transport',
            'transit': 'Transport',
            'commute': 'Transport',
            'parking': 'Transport',
            'car': 'Transport',
            'vehicle': 'Transport',
            
            # Healthcare (maps to Other since it's not a main category)
            'healthcare': 'Other',
            'medical': 'Other',
            'doctor': 'Other',
            'hospital': 'Other',
            'pharmacy': 'Other',
            'dental': 'Other'
        }
        
        # First, check for exact matches in the category map
        for key, value in category_map.items():
            if key == transaction_lower:
                print(f"DEBUG: Exact match found: '{key}' -> '{value}'")
                return value
        
        # Second, check for partial matches (substring)
        for key, value in category_map.items():
            if key in transaction_lower:
                print(f"DEBUG: Partial match found: '{key}' in '{transaction_lower}' -> '{value}'")
                return value
        
        # Third, check if it matches any budget category exactly (case-insensitive)
        for budget_category in self.default_budgets.keys():
            if budget_category.lower() == transaction_lower:
                print(f"DEBUG: Direct budget category match: '{transaction_lower}' -> '{budget_category}'")
                return budget_category
        
        print(f"DEBUG: No match found for '{transaction_category}', defaulting to 'Other'")
        return 'Other'

    def calculate_budget_progress(self, user_id, month=None, year=None):
        """Calculate budget usage for each category - FIXED"""
        try:
            # Get user-specific budgets (now includes ALL categories)
            user_budgets = self.get_user_budgets(user_id)
            
            # Get all expense transactions
            transactions = Transaction.query.filter_by(user_id=user_id, type='expense').all()
            
            if not transactions:
                print("DEBUG: No expense transactions found for user")
                return {}
            
            # Use current month/year if not specified
            if month is None:
                month = datetime.now().month
            if year is None:
                year = datetime.now().year
            
            print(f"DEBUG: Calculating budget for {month}/{year}")
            print(f"DEBUG: User budgets: {user_budgets}")
            print(f"DEBUG: Found {len(transactions)} expense transactions")
            
            # Initialize budget progress for ALL categories
            budget_progress = {}
            
            # Initialize totals for ALL budget categories
            category_totals = {}
            category_counts = {}
            
            for budget_category in user_budgets.keys():
                category_totals[budget_category] = 0.0
                category_counts[budget_category] = 0
            
            # Process each transaction
            for transaction in transactions:
                # Check if transaction is in the target month/year
                trans_date = transaction.date
                if hasattr(trans_date, 'month') and hasattr(trans_date, 'year'):
                    if trans_date.month == month and trans_date.year == year:
                        matched_category = self.match_category(transaction.category)
                        amount = float(transaction.amount)
                        
                        print(f"DEBUG: Processing transaction: '{transaction.category}' -> '{matched_category}' (${amount})")
                        
                        # Add to the appropriate budget category
                        if matched_category in category_totals:
                            category_totals[matched_category] += amount
                            category_counts[matched_category] += 1
                            print(f"DEBUG: Added to {matched_category}")
                        else:
                            # If category doesn't exist in budgets, add to Other
                            if 'Other' in category_totals:
                                category_totals['Other'] += amount
                                category_counts['Other'] += 1
                                print(f"DEBUG: Category '{matched_category}' not found, added to 'Other'")
                            else:
                                print(f"DEBUG: Category '{matched_category}' not found and 'Other' not available")
            
            # Calculate progress for each budget category
            total_spent_all = 0
            for budget_category, budget_amount in user_budgets.items():
                spent = category_totals.get(budget_category, 0)
                total_spent_all += spent
                transaction_count = category_counts.get(budget_category, 0)
                
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
                    'transaction_count': transaction_count
                }
                
                print(f"DEBUG: {budget_category}: ${spent} of ${budget_amount} ({progress}%) - {transaction_count} transactions")
            
            print(f"DEBUG: Total spent across all categories: ${total_spent_all}")
            print(f"DEBUG: Budget progress categories: {list(budget_progress.keys())}")
            
            return budget_progress
            
        except Exception as e:
            print(f"Budget calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def check_budget_alerts(self, user_id):
        """Check for budget alerts"""
        alerts = []
        progress = self.calculate_budget_progress(user_id)
        
        for category, data in progress.items():
            if data['progress'] >= 100:
                alerts.append({
                    'type': 'over_budget',
                    'category': category,
                    'message': f"You've exceeded your {category} budget by ${data['spent'] - data['budget']:.2f}"
                })
            elif data['progress'] >= 80:
                alerts.append({
                    'type': 'near_limit', 
                    'category': category,
                    'message': f"You're close to your {category} budget limit (${data['remaining']:.2f} remaining)"
                })
        
        return alerts

    def get_spending_trends(self, user_id, months=6):
        """Show spending trends over time"""
        trends = {}
        for i in range(months):
            month = (datetime.now().month - i - 1) % 12 + 1
            year = datetime.now().year - ((datetime.now().month - i - 1) // 12)
            progress = self.calculate_budget_progress(user_id, month, year)
            trends[f"{year}-{month:02d}"] = progress
        return trends

@budget_bp.route('/budget')
@login_required
def budget_page():
    try:
        budget_manager = BudgetManager()
        
        # Get month/year from query parameters or use current
        month = request.args.get('month', type=int)
        year = request.args.get('year', type=int)
        
        # Use user-specific budgets instead of default ones
        user_budgets = budget_manager.get_user_budgets(current_user.id)
        
        budget_progress = budget_manager.calculate_budget_progress(
            current_user.id, 
            month=month, 
            year=year
        )
        
        # Get budget alerts
        budget_alerts = budget_manager.check_budget_alerts(current_user.id)
        
        print(f"DEBUG: Budget progress result keys: {list(budget_progress.keys())}")
        print(f"DEBUG: Budget alerts: {budget_alerts}")
        
        # Get all user transactions for info
        all_transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        expense_transactions = [t for t in all_transactions if t.type == 'expense']
        
        return render_template('budget.html', 
                             title='Budget Management',
                             budget_progress=budget_progress,
                             budget_alerts=budget_alerts,
                             default_budgets=user_budgets,
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
                             budget_alerts=[],
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
            'type': t.type,
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
        'expense_transactions': len([t for t in transactions if t.type == 'expense']),
        'current_month': datetime.now().month,
        'current_year': datetime.now().year,
        'transactions': transaction_data
    })

@budget_bp.route('/api/update-budget', methods=['POST'])
@login_required
def update_budget():
    """API to update user budgets - FIXED VERSION"""
    try:
        data = request.get_json()
        budget_manager = BudgetManager()
        
        print(f"DEBUG: Received budget update data: {data}")
        
        # Get current user budgets (with all defaults)
        current_budgets = budget_manager.get_user_budgets(current_user.id)
        print(f"DEBUG: Current budgets before update: {current_budgets}")
        
        # Update only the provided categories, keep existing ones
        updated_budgets = current_budgets.copy()
        for category, amount in data.items():
            updated_budgets[category] = float(amount)
        
        print(f"DEBUG: Updated budgets after merge: {updated_budgets}")
        
        # Save all budgets
        user_budget = UserBudget.query.filter_by(user_id=current_user.id).first()
        if not user_budget:
            user_budget = UserBudget(user_id=current_user.id, budgets={})
        
        user_budget.budgets = updated_budgets
        user_budget.updated_at = datetime.utcnow()
        
        db.session.add(user_budget)
        db.session.commit()
        
        print(f"DEBUG: Successfully saved budgets: {user_budget.budgets}")
        
        return jsonify({
            'success': True, 
            'message': 'Budgets updated successfully',
            'budgets': user_budget.budgets
        })
    except Exception as e:
        print(f"ERROR updating budgets: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@budget_bp.route('/api/budget-alerts')
@login_required
def get_budget_alerts():
    """API to get budget alerts"""
    budget_manager = BudgetManager()
    alerts = budget_manager.check_budget_alerts(current_user.id)
    return jsonify({'alerts': alerts})

@budget_bp.route('/api/spending-trends')
@login_required
def spending_trends():
    """API to get spending trends"""
    budget_manager = BudgetManager()
    trends = budget_manager.get_spending_trends(current_user.id)
    return jsonify({'trends': trends})

@budget_bp.route('/export-budget-report')
@login_required
def export_budget_report():
    """Export budget report as CSV"""
    budget_manager = BudgetManager()
    progress = budget_manager.calculate_budget_progress(current_user.id)
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Category', 'Budget', 'Spent', 'Remaining', 'Progress%', 'Status'])
    
    for category, data in progress.items():
        writer.writerow([
            category, data['budget'], data['spent'], 
            data['remaining'], data['progress'], data['status']
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=budget-report.csv"}
    )

@budget_bp.route('/category-debug')
@login_required
def category_debug():
    """Debug page to see how categories are being matched"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    budget_manager = BudgetManager()
    
    category_analysis = []
    for t in transactions:
        if t.type == 'expense':
            matched = budget_manager.match_category(t.category)
            category_analysis.append({
                'original_category': t.category,
                'matched_category': matched,
                'amount': t.amount,
                'date': str(t.date),
                'note': t.note
            })
    
    return jsonify({
        'total_transactions': len(transactions),
        'expense_transactions': len(category_analysis),
        'category_mapping': category_analysis
    })

@budget_bp.route('/fix-budgets')
@login_required
def fix_budgets():
    """Fix user budgets by ensuring all categories exist including Other"""
    budget_manager = BudgetManager()
    
    # Get or create user budget
    user_budget = UserBudget.query.filter_by(user_id=current_user.id).first()
    if user_budget:
        # Merge with default budgets (including Other)
        merged_budgets = budget_manager.default_budgets.copy()
        merged_budgets.update(user_budget.budgets)
        user_budget.budgets = merged_budgets
    else:
        # Create new budget with all defaults including Other
        user_budget = UserBudget(
            user_id=current_user.id, 
            budgets=budget_manager.default_budgets.copy()
        )
    
    db.session.add(user_budget)
    db.session.commit()
    
    return jsonify({
        'success': True, 
        'message': 'Budgets fixed! All categories including Other are now available.', 
        'budgets': user_budget.budgets
    })