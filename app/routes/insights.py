from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required, current_user
from core.database import Transaction
import logging
import pandas as pd

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
    
# Add this to your app/routes/insights.py
@insights_bp.route('/test-dataset')
@login_required
def test_dataset():
    try:
        from data_loader import RealDatasetLoader
        loader = RealDatasetLoader()
        transactions = loader.load_and_preprocess()
        
        return jsonify({
            'dataset_loaded': True,
            'transaction_count': len(transactions),
            'sample_transactions': transactions[:3]  # Show first 3
        })
    except Exception as e:
        return jsonify({
            'dataset_loaded': False,
            'error': str(e)
        })
    
@insights_bp.route('/responsible-ai')
@login_required
def responsible_ai_dashboard():
    """Responsible AI monitoring dashboard"""
    from agents.responsible_ai import ResponsibleAIAuditor
    from core.database import Transaction
    
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    if not transactions:
        return render_template('responsible_ai.html', 
                             title='Responsible AI Dashboard',
                             audit_report=None,
                             no_data=True)
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': t.date,
        'type': t.type,
        'category': t.category,
        'amount': t.amount,
        'note': t.note or ''
    } for t in transactions])
    
    # Run Responsible AI audit
    auditor = ResponsibleAIAuditor()
    ai_categories = df['category'].tolist()
    audit_report = auditor.audit_ai_decisions(df.to_dict('records'), ai_categories)
    
    return render_template('responsible_ai.html',
                         title='Responsible AI Dashboard',
                         audit_report=audit_report,
                         no_data=False)