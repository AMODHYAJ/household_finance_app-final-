from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
from flask_login import login_required, current_user
from app import db
from app.models.forms import TransactionForm
from core.database import Transaction
from datetime import datetime
import pandas as pd

transactions_bp = Blueprint('transactions', __name__)

@transactions_bp.route('/')
@transactions_bp.route('/dashboard')
@login_required
def dashboard():
    # Get transactions for the current user
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    if not transactions:
        return render_template('dashboard.html', title='Dashboard', 
                             transactions=[], metrics={}, show_welcome=True)
    
    # Convert to DataFrame for calculations
    df = pd.DataFrame([{
        'id': t.id,
        'date': t.date,
        'type': t.t_type,
        'category': t.category,
        'amount': t.amount,
        'note': t.note
    } for t in transactions])
    
    # Calculate metrics
    total_income = df[df['type'] == 'income']['amount'].sum()
    total_expense = df[df['type'] == 'expense']['amount'].sum()
    savings = total_income - total_expense
    savings_rate = (savings / total_income * 100) if total_income > 0 else 0
    
    metrics = {
        'total_income': total_income,
        'total_expense': total_expense,
        'savings': savings,
        'savings_rate': savings_rate,
        'transaction_count': len(transactions)
    }
    
    # Get recent transactions
    recent_transactions = Transaction.query.filter_by(user_id=current_user.id)\
        .order_by(Transaction.date.desc())\
        .limit(10)\
        .all()
    
    return render_template('dashboard.html', 
                         title='Dashboard',
                         metrics=metrics,
                         transactions=recent_transactions,
                         show_welcome=False)

@transactions_bp.route('/add-transaction', methods=['GET', 'POST'])
@login_required
def add_transaction():
    form = TransactionForm()
    
    if form.validate_on_submit():
        transaction = Transaction(
            user_id=current_user.id,
            t_type=form.t_type.data,
            category=form.category.data,
            amount=form.amount.data,
            date=form.date.data,
            note=form.note.data or None
        )
        
        db.session.add(transaction)
        db.session.commit()
        
        flash('Transaction added successfully!', 'success')
        return redirect(url_for('transactions.dashboard'))
    
    return render_template('add_transaction.html', title='Add Transaction', form=form)

@transactions_bp.route('/view-transactions')
@login_required
def view_transactions():
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    transactions = Transaction.query.filter_by(user_id=current_user.id)\
        .order_by(Transaction.date.desc())\
        .paginate(page=page, per_page=per_page)
    
    return render_template('view_transactions.html', 
                         title='View Transactions',
                         transactions=transactions)

@transactions_bp.route('/api/transactions')
@login_required
def api_transactions():
    """API endpoint for transactions data (used by charts)"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    data = [{
        'id': t.id,
        'date': t.date.isoformat(),
        'type': t.t_type,
        'category': t.category,
        'amount': t.amount,
        'note': t.note
    } for t in transactions]
    
    return jsonify(data)