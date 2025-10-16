from flask import Blueprint, render_template, jsonify, current_app, request
from flask_login import login_required, current_user
import pandas as pd
from core.database import Transaction
import json

charts_bp = Blueprint('charts', __name__)

@charts_bp.route('/charts')
@login_required
def charts_page():
    """Basic charts page"""
    return render_template('charts.html', title='Financial Charts')

@charts_bp.route('/api/chart-data')
@login_required
def chart_data():
    """API endpoint for charts with proper data aggregation"""
    try:
        transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        
        print(f"📊 User {current_user.id} has {len(transactions)} transactions")
        
        if not transactions:
            print("❌ No transactions found for user")
            return jsonify({'error': 'No transactions available'})
        
        chart_agent = current_app.architect_agent.chart_agent
        
        # Generate all charts
        print("🔄 Generating expenses pie chart...")
        expenses_pie_chart = chart_agent.create_expenses_by_category_chart(transactions)
        
        print("🔄 Generating expenses bar chart...")
        expenses_bar_chart = chart_agent.create_expenses_bar_chart(transactions)
        
        print("🔄 Generating category comparison chart...")
        category_comparison_chart = chart_agent.create_category_comparison_chart(transactions)
        
        print("🔄 Generating income vs expenses chart...")
        income_expense_chart = chart_agent.create_income_vs_expenses_chart(transactions)
        
        print("🔄 Generating spending timeline chart...")
        spending_timeline_chart = chart_agent.create_spending_timeline_chart(transactions)
        
        # Convert to JSON
        chart_data = {}
        
        charts = [
            ('expenses_pie_chart', expenses_pie_chart),
            ('expenses_bar_chart', expenses_bar_chart),
            ('category_comparison', category_comparison_chart),
            ('income_vs_expenses', income_expense_chart),
            ('spending_timeline', spending_timeline_chart)
        ]
        
        for chart_name, chart in charts:
            if chart:
                chart_data[chart_name] = chart.to_json()
                print(f"✅ {chart_name} generated")
            else:
                chart_data[chart_name] = None
                print(f"❌ {chart_name} generation failed")
        
        print("🎉 All charts processed successfully")
        return jsonify(chart_data)
        
    except Exception as e:
        print(f"❌ Chart data endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate charts: {str(e)}'})
    
@charts_bp.route('/ai-charts')
@login_required
def ai_charts_page():
    """AI-powered chart generation page"""
    return render_template('ai_charts.html', title='AI-Powered Charts')

@charts_bp.route('/api/ai-charts', methods=['POST'])
@login_required
def generate_ai_chart():
    """API endpoint for AI chart generation"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Please enter a chart request'})
        
        # Get user transactions
        transactions = Transaction.query.filter_by(user_id=current_user.id).all()
        
        if not transactions:
            return jsonify({
                'success': False,
                'error': 'No transactions found. Add some transactions to generate charts.',
                'chart_type': 'no_data',
                'analysis_notes': 'No transaction data available',
                'insights': [
                    'Start by adding your first transaction',
                    'Use the "Add Transaction" button to begin',
                    'Once you have data, you can generate various charts'
                ]
            })
        
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
        print(f"AI chart generation failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Chart generation failed: {str(e)}',
            'chart_type': 'error',
            'analysis_notes': 'System error occurred',
            'insights': ['Please try a different query or try again later']
        })
    
@charts_bp.route('/debug-pie-data')
@login_required
def debug_pie_data():
    """Debug the actual pie chart data"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    chart_agent = current_app.architect_agent.chart_agent
    pie_chart = chart_agent.create_expenses_by_category_chart(transactions)
    
    # Get the actual data from the chart and convert ndarrays to lists
    if pie_chart and pie_chart.data:
        trace = pie_chart.data[0]
        debug_info = {
            'labels': trace.labels.tolist() if hasattr(trace.labels, 'tolist') else list(trace.labels),
            'values': trace.values.tolist() if hasattr(trace.values, 'tolist') else list(trace.values),
            'values_sum': sum(trace.values),
            'percentages': [f"{(v/sum(trace.values))*100:.1f}%" for v in trace.values]
        }
    else:
        debug_info = {'error': 'No chart data'}
    
    return jsonify(debug_info)

@charts_bp.route('/debug-timeline-data')
@login_required
def debug_timeline_data():
    """Debug the timeline chart data"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    df_data = []
    for t in transactions:
        transaction_type = getattr(t, 'type', None) or getattr(t, 't_type', 'expense')
        if str(transaction_type).lower() == 'expense':
            df_data.append({
                'date': t.date,
                'amount': float(t.amount) if t.amount else 0.0,
                'category': t.category,
                'note': t.note
            })
    
    df = pd.DataFrame(df_data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        daily_expenses = df.groupby('date')['amount'].sum().reset_index()
        daily_expenses = daily_expenses.sort_values('date')
        
        debug_info = {
            'total_days': len(daily_expenses),
            'daily_totals': daily_expenses.to_dict('records'),
            'total_expenses': daily_expenses['amount'].sum(),
            'max_daily': daily_expenses['amount'].max(),
            'min_daily': daily_expenses['amount'].min()
        }
    else:
        debug_info = {'error': 'No expense data'}
    
    return jsonify(debug_info)