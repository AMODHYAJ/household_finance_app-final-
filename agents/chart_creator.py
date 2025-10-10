import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

class ChartCreatorAgent:
    def __init__(self, data_agent):
        self.data_agent = data_agent

    def create_expenses_by_category_chart(self, transactions):
        """Create expenses by category pie chart"""
        if not transactions:
            return self._create_empty_chart("No data available")
        
        df = pd.DataFrame([{
            'category': t.category,
            'amount': t.amount,
            'type': t.t_type
        } for t in transactions])
        
        expenses = df[df['type'] == 'expense']
        if expenses.empty:
            return self._create_empty_chart("No expense data available")
        
        category_totals = expenses.groupby('category')['amount'].sum()
        
        fig = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title="Expenses by Category"
        )
        
        return fig

    def create_income_vs_expenses_chart(self, transactions):
        """Create income vs expenses bar chart"""
        if not transactions:
            return self._create_empty_chart("No data available")
        
        # Convert to monthly data
        df = pd.DataFrame([{
            'date': t.date,
            'type': t.t_type,
            'amount': t.amount
        } for t in transactions])
        
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly_data = df.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure()
        if 'income' in monthly_data.columns:
            fig.add_trace(go.Bar(name='Income', x=monthly_data.index, y=monthly_data['income'], marker_color='green'))
        if 'expense' in monthly_data.columns:
            fig.add_trace(go.Bar(name='Expense', x=monthly_data.index, y=monthly_data['expense'], marker_color='red'))
        
        fig.update_layout(title="Monthly Income vs Expenses", barmode='group')
        return fig

    def _create_empty_chart(self, message):
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title=message)
        return fig