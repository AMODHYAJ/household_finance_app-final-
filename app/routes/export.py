from flask import Blueprint, send_file, current_app
from flask_login import login_required, current_user
from core.database import Transaction
import pandas as pd
import io
from datetime import datetime

export_bp = Blueprint('export', __name__)

@export_bp.route('/export/csv')
@login_required
def export_csv():
    """Export transactions as CSV"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    if not transactions:
        return "No transactions to export", 400
    
    data = [{
        'Date': t.date.strftime('%Y-%m-%d'),
        'Type': t.type.title(),
        'Category': t.category,
        'Amount': t.amount,
        'Note': t.note or '',
        'Created': t.created_at.strftime('%Y-%m-%d %H:%M:%S') if t.created_at else ''
    } for t in transactions]
    
    df = pd.DataFrame(data)
    
    # Create CSV in memory
    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    filename = f"financial_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return send_file(
        output,
        mimetype='text/csv',
        as_attachment=True,
        download_name=filename
    )

@export_bp.route('/export/summary')
@login_required
def export_summary():
    """Export financial summary as text file"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    if not transactions:
        return "No data to export", 400
    
    # Generate text summary
    summary_lines = []
    summary_lines.append("FINANCIAL SUMMARY REPORT")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"User: {current_user.username}")
    summary_lines.append("")
    
    df = pd.DataFrame([{
        'date': t.date,
        'type': t.type,
        'category': t.category,
        'amount': t.amount
    } for t in transactions])
    
    total_income = df[df['type'] == 'income']['amount'].sum()
    total_expense = df[df['type'] == 'expense']['amount'].sum()
    net_savings = total_income - total_expense
    
    summary_lines.append("SUMMARY STATISTICS:")
    summary_lines.append(f"Total Income: ${total_income:,.2f}")
    summary_lines.append(f"Total Expenses: ${total_expense:,.2f}")
    summary_lines.append(f"Net Savings: ${net_savings:,.2f}")
    summary_lines.append(f"Transaction Count: {len(transactions)}")
    summary_lines.append("")
    
    # Top categories
    if not df[df['type'] == 'expense'].empty:
        top_categories = df[df['type'] == 'expense'].groupby('category')['amount'].sum().nlargest(5)
        summary_lines.append("TOP EXPENSE CATEGORIES:")
        for category, amount in top_categories.items():
            summary_lines.append(f"  {category}: ${amount:,.2f}")
    
    # Convert to text file
    summary_text = "\n".join(summary_lines)
    output = io.BytesIO()
    output.write(summary_text.encode('utf-8'))
    output.seek(0)
    
    filename = f"financial_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    return send_file(
        output,
        mimetype='text/plain',
        as_attachment=True,
        download_name=filename
    )