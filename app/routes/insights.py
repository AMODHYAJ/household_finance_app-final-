from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required, current_user
from core.database import Transaction
import logging

insights_bp = Blueprint('insights', __name__)

@insights_bp.route('/insights')
@login_required
def insights_page():
    # Get transactions data
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    print(f"DEBUG: Found {len(transactions)} transactions for user {current_user.id}")
    
    if not transactions:
        print("DEBUG: No transactions found, showing empty state")
        return render_template('insights.html', 
                             title='AI Insights',
                             insights={},
                             no_data=True)
    
    # Generate insights using our simplified agent
    try:
        insight_agent = current_app.architect_agent.insight_agent
        print("DEBUG: Insight agent accessed successfully")
        
        insights = insight_agent.generate_all(transactions)
        print(f"DEBUG: Generated insights: {insights}")
        
        return render_template('insights.html',
                             title='AI Insights',
                             insights=insights,
                             no_data=False)
    except Exception as e:
        print(f"DEBUG: Error generating insights: {str(e)}")
        return render_template('insights.html',
                             title='AI Insights',
                             insights={},
                             no_data=True,
                             error=str(e))

@insights_bp.route('/api/insights-data')
@login_required
def insights_data():
    """API endpoint for insights data"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    if not transactions:
        return jsonify({'error': 'No data available'})
    
    try:
        insight_agent = current_app.architect_agent.insight_agent
        insights = insight_agent.generate_all(transactions)
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)})