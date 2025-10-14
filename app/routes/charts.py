from flask import Blueprint, render_template, jsonify, current_app, request
from flask_login import login_required, current_user
from core.database import Transaction
import json
import logging

charts_bp = Blueprint('charts', __name__)

@charts_bp.route('/charts')
@login_required
def charts_page():
    return render_template('charts.html', title='Charts & Visualizations')

@charts_bp.route('/api/chart-data')
@login_required
def chart_data():
    """API endpoint for chart data"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    if not transactions:
        return jsonify({'error': 'No data available'})
    
    # Get chart data using the agent
    chart_agent = current_app.architect_agent.chart_agent
    
    # Convert transactions to the format expected by chart agent
    expenses_chart = chart_agent.create_expenses_by_category_chart(transactions)
    income_expense_chart = chart_agent.create_income_vs_expenses_chart(transactions)
    
    # Convert charts to JSON
    chart_data = {
        'expenses_by_category': expenses_chart.to_json() if expenses_chart else None,
        'income_vs_expenses': income_expense_chart.to_json() if income_expense_chart else None
    }
    
    return jsonify(chart_data)

@charts_bp.route('/ai-charts', methods=['GET', 'POST'])
@login_required
def ai_charts():
    """AI-powered chart generation with enhanced error handling"""
    if request.method == 'POST':
        try:
            data = request.json
            query = data.get('query', '').strip()
            
            if not query:
                return jsonify({'error': 'Please enter a chart request'})
            
            # Get user transactions
            transactions = Transaction.query.filter_by(user_id=current_user.id).all()
            
            if not transactions:
                return jsonify({'error': 'No transactions found. Add some transactions to generate charts.'})
            
            transaction_data = [{
                'id': t.id,
                'date': t.date.isoformat() if t.date else None,
                'type': t.type,
                'category': t.category,
                'amount': float(t.amount),
                'note': t.note or ''
            } for t in transactions]
            
            # Generate chart using LLM
            from agents.llm_chart_generator import LLMChartGenerator
            chart_generator = LLMChartGenerator()
            chart_result = chart_generator.generate_chart_from_query(query, transaction_data)
            
            return jsonify(chart_result)
            
        except Exception as e:
            logging.error(f"AI chart generation failed: {e}")
            return jsonify({'error': f'Chart generation failed: {str(e)}'})
    
    return render_template('ai_charts.html', title='AI-Powered Charts')