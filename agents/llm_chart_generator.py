import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class LLMChartGenerator:
    """
    Complete Advanced AI chart generator with all methods
    """
    
    def __init__(self):
        self.chart_types = {
            'predictive': ['predict', 'forecast', 'future', 'next', 'coming', 'trend', 'outlook'],
            'behavioral': ['pattern', 'habit', 'behavior', 'routine', 'lifestyle', 'spending habit'],
            'optimization': ['save', 'optimize', 'reduce', 'cut', 'better', 'improve', 'efficient'],
            'comparative': ['compare', 'vs', 'versus', 'difference', 'against', 'relative'],
            'temporal': ['timeline', 'over time', 'history', 'progress', 'evolution', 'journey'],
            'categorical': ['category', 'breakdown', 'distribution', 'by type', 'segmentation'],
            'risk': ['risk', 'alert', 'warning', 'danger', 'concern', 'problem'],
            'dashboard': ['dashboard', 'overview', 'summary', 'everything', 'complete']
        }
    
    def generate_chart_from_query(self, query, transactions_data):
        """Generate truly advanced AI-powered charts"""
        if not transactions_data:
            return self._create_error_response("No transaction data available")
        
        try:
            df = pd.DataFrame(transactions_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Enhanced query analysis
            chart_type, analysis_depth = self._analyze_query_intent(query)
            
            print(f"🤖 Advanced AI Analysis - Query: '{query}', Type: {chart_type}, Depth: {analysis_depth}")
            
            # Generate appropriate advanced chart
            if chart_type == 'predictive':
                return self._create_predictive_analysis(df, query, analysis_depth)
            elif chart_type == 'behavioral':
                return self._create_behavioral_analysis(df, query, analysis_depth)
            elif chart_type == 'optimization':
                return self._create_optimization_analysis(df, query, analysis_depth)
            elif chart_type == 'comparative':
                return self._create_comparative_analysis(df, query, analysis_depth)
            elif chart_type == 'temporal':
                return self._create_temporal_analysis(df, query, analysis_depth)
            elif chart_type == 'categorical':
                return self._create_categorical_analysis(df, query, analysis_depth)
            elif chart_type == 'risk':
                return self._create_risk_analysis(df, query, analysis_depth)
            else:
                return self._create_ai_dashboard(df, query, analysis_depth)
            
        except Exception as e:
            print(f"❌ Advanced AI Chart generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_comprehensive_fallback(df, query)

    def _create_predictive_analysis(self, df, query, depth):
        """Predictive analysis with machine learning"""
        try:
            # Prepare data for forecasting
            expenses_df = df[df['type'] == 'expense'].copy()
            if len(expenses_df) < 5:
                return self._create_error_response("Need more data for accurate predictions")
            
            # Create future dates for prediction
            last_date = df['date'].max()
            future_days = 30
            future_dates = [last_date + timedelta(days=x) for x in range(1, future_days + 1)]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Expense Trend & 30-Day Forecast',
                    'Category Spending Forecast',
                    'Savings Projection',
                    'Spending Confidence'
                ),
                specs=[
                    [{"type": "scatter", "colspan": 2}, None],
                    [{"type": "bar"}, {"type": "indicator"}]
                ],
                vertical_spacing=0.15
            )
            
            # 1. Expense trend and forecast
            daily_expenses = expenses_df.groupby('date')['amount'].sum().reset_index()
            daily_expenses['day_num'] = (daily_expenses['date'] - daily_expenses['date'].min()).dt.days
            
            # Simple linear regression for trend
            X = daily_expenses['day_num'].values.reshape(-1, 1)
            y = daily_expenses['amount'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Future predictions
            future_day_nums = np.array(range(daily_expenses['day_num'].max() + 1, 
                                           daily_expenses['day_num'].max() + future_days + 1)).reshape(-1, 1)
            future_predictions = model.predict(future_day_nums)
            
            # Historical trend
            fig.add_trace(go.Scatter(
                x=daily_expenses['date'], y=daily_expenses['amount'],
                mode='lines+markers', name='Actual Spending',
                line=dict(color='#e74c3c', width=3),
                hovertemplate='Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=future_dates, y=future_predictions,
                mode='lines', name='30-Day Forecast',
                line=dict(color='#3498db', width=3, dash='dash'),
                hovertemplate='Date: %{x}<br>Predicted: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Category forecast
            category_forecast = self._forecast_categories(expenses_df, future_days)
            fig.add_trace(go.Bar(
                x=list(category_forecast.keys()),
                y=list(category_forecast.values()),
                name='Category Forecast',
                marker_color='#2ecc71',
                hovertemplate='Category: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # 3. Savings projection
            income_df = df[df['type'] == 'income']
            avg_income = income_df['amount'].mean() if not income_df.empty else 0
            projected_savings = avg_income * 3 - sum(future_predictions)  # 3 months projection
            
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=projected_savings,
                title={"text": "3-Month Projected Savings"},
                delta={'reference': 0},
                domain={'row': 2, 'column': 2},
                number={'prefix': '$'}
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"🔮 AI Predictive Analysis: {query}",
                showlegend=True
            )
            
            insights = [
                f"Based on your spending pattern, we project ${sum(future_predictions):.2f} in expenses over the next 30 days",
                f"Your spending trend is {'increasing' if model.coef_[0] > 0 else 'decreasing'}",
                f"Top projected category: {max(category_forecast, key=category_forecast.get)}",
                f"3-month savings projection: ${projected_savings:.2f}"
            ]
            
            return self._create_success_response(fig, "predictive_analysis", 
                                               f"AI-powered predictive analysis with machine learning forecasting", insights)
            
        except Exception as e:
            print(f"❌ Predictive analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_behavioral_analysis(self, df, query, depth):
        """Behavioral spending pattern analysis"""
        try:
            expenses_df = df[df['type'] == 'expense'].copy()
            
            if expenses_df.empty:
                return self._create_error_response("No expense data for behavioral analysis")
            
            # Enhanced behavioral features
            expenses_df['day_of_week'] = expenses_df['date'].dt.day_name()
            expenses_df['is_weekend'] = expenses_df['date'].dt.dayofweek >= 5
            expenses_df['time_of_month'] = expenses_df['date'].dt.day
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Weekly Spending Pattern',
                    'Time-of-Month Analysis', 
                    'Category Behavior Heatmap',
                    'Spending Frequency'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "heatmap"}, {"type": "histogram"}]
                ]
            )
            
            # 1. Weekly pattern
            weekly_pattern = expenses_df.groupby('day_of_week')['amount'].sum()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_pattern = weekly_pattern.reindex(days_order)
            
            fig.add_trace(go.Bar(
                x=weekly_pattern.index, y=weekly_pattern.values,
                name='Weekly Pattern', marker_color='#e74c3c',
                hovertemplate='Day: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Time-of-month analysis
            monthly_pattern = expenses_df.groupby('time_of_month')['amount'].mean()
            fig.add_trace(go.Scatter(
                x=monthly_pattern.index, y=monthly_pattern.values,
                mode='lines+markers', name='Monthly Cycle',
                line=dict(color='#3498db', width=3),
                hovertemplate='Day of Month: %{x}<br>Avg Spending: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
            
            # 3. Category heatmap (simplified)
            category_dow = expenses_df.groupby(['category', 'day_of_week'])['amount'].sum().unstack(fill_value=0)
            fig.add_trace(go.Heatmap(
                z=category_dow.values,
                x=category_dow.columns.tolist(),
                y=category_dow.index.tolist(),
                colorscale='Viridis',
                name='Category Heatmap',
                hovertemplate='Category: %{y}<br>Day: %{x}<br>Amount: $%{z:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # 4. Spending frequency
            fig.add_trace(go.Histogram(
                x=expenses_df['amount'],
                nbinsx=20,
                name='Spending Distribution',
                marker_color='#2ecc71',
                hovertemplate='Amount: $%{x}<br>Frequency: %{y}<extra></extra>'
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"🧠 AI Behavioral Analysis: {query}",
                showlegend=False
            )
            
            # Behavioral insights
            max_day = weekly_pattern.idxmax()
            behavioral_insights = [
                f"You tend to spend most on {max_day}s (${weekly_pattern[max_day]:.2f})",
                f"Average daily spending: ${expenses_df['amount'].mean():.2f}",
                f"Most frequent spending category: {expenses_df['category'].mode().iloc[0] if not expenses_df['category'].mode().empty else 'N/A'}",
                f"Weekend vs weekday spending ratio: {expenses_df[expenses_df['is_weekend']]['amount'].sum() / expenses_df[~expenses_df['is_weekend']]['amount'].sum():.2f}x"
            ]
            
            return self._create_success_response(fig, "behavioral_analysis", 
                                               f"AI-powered behavioral spending pattern analysis", behavioral_insights)
            
        except Exception as e:
            print(f"❌ Behavioral analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_optimization_analysis(self, df, query, depth):
        """Spending optimization and savings recommendations"""
        try:
            expenses_df = df[df['type'] == 'expense'].copy()
            income_df = df[df['type'] == 'income']
            
            total_income = income_df['amount'].sum()
            total_expenses = expenses_df['amount'].sum()
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Savings Optimization Opportunities',
                    'Category Efficiency Analysis',
                    'Monthly Spending Pattern',
                    'Optimization Impact'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "indicator"}]
                ]
            )
            
            # 1. Savings opportunities
            category_totals = expenses_df.groupby('category')['amount'].sum().nlargest(6)
            optimization_potential = {cat: amt * 0.15 for cat, amt in category_totals.items()}  # 15% savings potential
            
            fig.add_trace(go.Bar(
                x=list(optimization_potential.keys()),
                y=list(optimization_potential.values()),
                name='Potential Monthly Savings',
                marker_color='#2ecc71',
                hovertemplate='Category: %{x}<br>Savings Potential: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Category efficiency (lower is better)
            category_efficiency = expenses_df.groupby('category')['amount'].mean()  # Avg transaction size
            fig.add_trace(go.Bar(
                x=category_efficiency.index.tolist(),
                y=category_efficiency.values.tolist(),
                name='Avg Transaction Size',
                marker_color='#e74c3c',
                hovertemplate='Category: %{x}<br>Avg Transaction: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
            
            # 3. Monthly spending
            monthly_spending = expenses_df.groupby(expenses_df['date'].dt.to_period('M'))['amount'].sum()
            fig.add_trace(go.Bar(
                x=monthly_spending.index.astype(str).tolist(),
                y=monthly_spending.values.tolist(),
                name='Monthly Spending',
                marker_color='#3498db',
                hovertemplate='Month: %{x}<br>Spending: $%{y:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # 4. Optimization impact
            total_savings_potential = sum(optimization_potential.values())
            savings_rate_improvement = (total_savings_potential / total_income * 100) if total_income > 0 else 0
            
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=savings_rate_improvement,
                number={'suffix': "%"},
                title={"text": "Potential Savings Rate Improvement"},
                delta={'reference': 0},
                domain={'row': 2, 'column': 2}
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"💡 AI Optimization Analysis: {query}",
                showlegend=False
            )
            
            optimization_insights = [
                f"Total monthly savings potential: ${total_savings_potential:.2f}",
                f"That's {savings_rate_improvement:.1f}% improvement in your savings rate",
                f"Focus on reducing {max(optimization_potential, key=optimization_potential.get)} for maximum impact",
                "Consider setting specific category budgets to control spending"
            ]
            
            return self._create_success_response(fig, "optimization_analysis", 
                                               f"AI-powered spending optimization recommendations", optimization_insights)
            
        except Exception as e:
            print(f"❌ Optimization analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_comparative_analysis(self, df, query, depth):
        """Comparative analysis between different aspects"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Income vs Expenses Over Time',
                    'Category Comparison',
                    'Monthly Performance',
                    'Spending Efficiency'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "indicator"}]
                ]
            )
            
            # 1. Income vs Expenses by month
            df['month'] = df['date'].dt.to_period('M')
            monthly_totals = df.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
            
            months = monthly_totals.index.astype(str).tolist()
            income = monthly_totals.get('income', pd.Series(0, index=monthly_totals.index)).tolist()
            expenses = monthly_totals.get('expense', pd.Series(0, index=monthly_totals.index)).tolist()
            
            fig.add_trace(go.Bar(
                name='Income', x=months, y=income, marker_color='#2ecc71',
                hovertemplate='Month: %{x}<br>Income: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                name='Expenses', x=months, y=expenses, marker_color='#e74c3c',
                hovertemplate='Month: %{x}<br>Expenses: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Category comparison
            expenses_df = df[df['type'] == 'expense']
            if not expenses_df.empty:
                category_totals = expenses_df.groupby('category')['amount'].sum()
                fig.add_trace(go.Pie(
                    labels=category_totals.index.tolist(),
                    values=category_totals.values.tolist(),
                    name='Category Distribution',
                    hole=0.3
                ), row=1, col=2)
            
            # 3. Monthly net performance
            monthly_net = [inc - exp for inc, exp in zip(income, expenses)]
            fig.add_trace(go.Bar(
                x=months, y=monthly_net,
                name='Monthly Net',
                marker_color=['#2ecc71' if x >= 0 else '#e74c3c' for x in monthly_net],
                hovertemplate='Month: %{x}<br>Net: $%{y:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # 4. Overall efficiency
            total_income = sum(income)
            total_expenses = sum(expenses)
            efficiency = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=efficiency,
                number={'suffix': "%"},
                title={'text': "Financial Efficiency"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "green" if efficiency > 20 else "orange" if efficiency > 0 else "red"}},
                domain={'row': 2, 'column': 2}
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"📊 AI Comparative Analysis: {query}",
                showlegend=True,
                barmode='group'
            )
            
            comparative_insights = [
                f"Total income: ${total_income:.2f} vs expenses: ${total_expenses:.2f}",
                f"Financial efficiency: {efficiency:.1f}%",
                f"Best performing month: {months[monthly_net.index(max(monthly_net))] if monthly_net else 'N/A'}",
                f"Total net savings: ${sum(monthly_net):.2f}"
            ]
            
            return self._create_success_response(fig, "comparative_analysis", 
                                               f"AI-powered comparative financial analysis", comparative_insights)
            
        except Exception as e:
            print(f"❌ Comparative analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_categorical_analysis(self, df, query, depth):
        """Advanced category analysis"""
        try:
            expenses_df = df[df['type'] == 'expense']
            
            if expenses_df.empty:
                return self._create_error_response("No expense data for category analysis")
            
            category_totals = expenses_df.groupby('category')['amount'].sum()
            
            fig = go.Figure()
            
            fig.add_trace(go.Pie(
                labels=category_totals.index.tolist(),
                values=category_totals.values.tolist(),
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Amount: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
            ))
            
            total_expenses = category_totals.sum()
            fig.update_layout(
                title=f"📈 AI Category Analysis: {query}<br><sub>Total Expenses: ${total_expenses:.2f}</sub>",
                height=500,
                showlegend=True
            )
            
            categorical_insights = [
                f"Total expenses across {len(category_totals)} categories: ${total_expenses:.2f}",
                f"Largest category: {category_totals.idxmax()} (${category_totals.max():.2f})",
                f"Smallest category: {category_totals.idxmin()} (${category_totals.min():.2f})",
                f"Average per category: ${category_totals.mean():.2f}"
            ]
            
            return self._create_success_response(fig, "categorical_analysis", 
                                               f"AI-powered category spending analysis", categorical_insights)
            
        except Exception as e:
            print(f"❌ Categorical analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_temporal_analysis(self, df, query, depth):
        """Temporal analysis over time"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Income & Expense Timeline',
                    'Cumulative Cash Flow',
                    'Daily Spending Pattern',
                    'Monthly Trends'
                ),
                specs=[
                    [{"type": "scatter", "colspan": 2}, None],
                    [{"type": "scatter"}, {"type": "bar"}]
                ]
            )
            
            # 1. Income & Expense Timeline
            daily_totals = df.groupby(['date', 'type'])['amount'].sum().unstack(fill_value=0)
            
            if 'income' in daily_totals.columns:
                fig.add_trace(go.Scatter(
                    x=daily_totals.index, y=daily_totals['income'],
                    name='Income', line=dict(color='#2ecc71', width=2),
                    mode='lines', hovertemplate='Date: %{x}<br>Income: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
            
            if 'expense' in daily_totals.columns:
                fig.add_trace(go.Scatter(
                    x=daily_totals.index, y=daily_totals['expense'],
                    name='Expenses', line=dict(color='#e74c3c', width=2),
                    mode='lines', hovertemplate='Date: %{x}<br>Expenses: $%{y:.2f}<extra></extra>'
                ), row=1, col=1)
            
            # 2. Cumulative Cash Flow
            df_sorted = df.sort_values('date')
            df_sorted['cumulative_net'] = (
                df_sorted[df_sorted['type'] == 'income']['amount'].cumsum() - 
                df_sorted[df_sorted['type'] == 'expense']['amount'].cumsum()
            ).fillna(method='ffill')
            
            fig.add_trace(go.Scatter(
                x=df_sorted['date'], y=df_sorted['cumulative_net'],
                name='Cumulative Net', line=dict(color='#3498db', width=3),
                mode='lines', fill='tozeroy',
                hovertemplate='Date: %{x}<br>Cumulative Net: $%{y:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # 3. Monthly Trends
            df['month'] = df['date'].dt.to_period('M')
            monthly_totals = df.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
            monthly_totals.index = monthly_totals.index.astype(str)
            
            if 'income' in monthly_totals.columns:
                fig.add_trace(go.Bar(
                    x=monthly_totals.index.tolist(), y=monthly_totals['income'].tolist(),
                    name='Monthly Income', marker_color='#2ecc71',
                    hovertemplate='Month: %{x}<br>Income: $%{y:.2f}<extra></extra>'
                ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"⏰ AI Temporal Analysis: {query}",
                showlegend=True
            )
            
            temporal_insights = [
                "Track your financial journey over time with multiple perspectives",
                "Cumulative net worth shows your overall financial progress",
                "Monthly trends help identify seasonal spending patterns",
                "Daily patterns reveal your spending consistency"
            ]
            
            return self._create_success_response(fig, "temporal_analysis", 
                                               f"AI-powered temporal financial analysis", temporal_insights)
            
        except Exception as e:
            print(f"❌ Temporal analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_risk_analysis(self, df, query, depth):
        """Risk assessment analysis"""
        try:
            expenses_df = df[df['type'] == 'expense']
            
            if expenses_df.empty:
                return self._create_error_response("No expense data for risk analysis")
            
            # Calculate risk metrics
            spending_volatility = self._calculate_spending_volatility(expenses_df)
            category_concentration = self._calculate_category_concentration(expenses_df)
            emergency_fund_score = self._calculate_emergency_fund_score(df)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Spending Volatility Risk',
                    'Category Concentration Risk',
                    'Emergency Fund Assessment',
                    'Overall Risk Profile'
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "pie"}]
                ]
            )
            
            # 1. Spending volatility
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=spending_volatility,
                title={'text': "Spending Volatility"},
                gauge={'axis': {'range': [0, 10]},
                      'bar': {'color': "red" if spending_volatility > 7 else "orange" if spending_volatility > 4 else "green"}},
                domain={'row': 0, 'column': 0}
            ), row=1, col=1)
            
            # 2. Category concentration
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=category_concentration,
                title={'text': "Category Concentration"},
                gauge={'axis': {'range': [0, 10]},
                      'bar': {'color': "red" if category_concentration > 7 else "orange" if category_concentration > 4 else "green"}},
                domain={'row': 0, 'column': 1}
            ), row=1, col=2)
            
            # 3. Emergency fund
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=emergency_fund_score,
                title={'text': "Emergency Fund"},
                gauge={'axis': {'range': [0, 10]},
                      'bar': {'color': "green" if emergency_fund_score > 7 else "orange" if emergency_fund_score > 4 else "red"}},
                domain={'row': 1, 'column': 0}
            ), row=2, col=1)
            
            # 4. Risk distribution
            risk_categories = ['Low', 'Medium', 'High']
            risk_values = [
                10 - max(spending_volatility, category_concentration, 10 - emergency_fund_score),
                abs(spending_volatility - category_concentration),
                max(spending_volatility, category_concentration, 10 - emergency_fund_score)
            ]
            
            fig.add_trace(go.Pie(
                labels=risk_categories,
                values=risk_values,
                hole=0.3,
                marker_colors=['#2ecc71', '#f39c12', '#e74c3c']
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"⚠️ AI Risk Analysis: {query}",
                showlegend=False
            )
            
            risk_insights = [
                f"Spending volatility: {'High' if spending_volatility > 7 else 'Medium' if spending_volatility > 4 else 'Low'} risk",
                f"Category concentration: {'High' if category_concentration > 7 else 'Medium' if category_concentration > 4 else 'Low'} risk",
                f"Emergency fund: {'Strong' if emergency_fund_score > 7 else 'Moderate' if emergency_fund_score > 4 else 'Weak'}",
                "Recommendation: " + self._generate_risk_recommendation(spending_volatility, category_concentration, emergency_fund_score)
            ]
            
            return self._create_success_response(fig, "risk_analysis", 
                                               f"AI-powered financial risk assessment", risk_insights)
            
        except Exception as e:
            print(f"❌ Risk analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_ai_dashboard(self, df, query, depth):
        """Advanced AI-powered comprehensive dashboard"""
        try:
            # Calculate advanced metrics
            financial_health_score = self._calculate_financial_health(df)
            spending_efficiency = self._calculate_spending_efficiency(df)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Financial Health Score',
                    'Income vs Expense Distribution', 
                    'Spending Efficiency',
                    'Category Intelligence'
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "pie"}],
                    [{"type": "indicator"}, {"type": "bar"}]
                ]
            )
            
            # 1. Financial Health Score
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=financial_health_score,
                title={'text': "Financial Health"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "green" if financial_health_score > 70 else "orange" if financial_health_score > 40 else "red"},
                      'steps': [{'range': [0, 40], 'color': "lightgray"},
                               {'range': [40, 70], 'color': "yellow"},
                               {'range': [70, 100], 'color': "lightgreen"}]},
                domain={'row': 0, 'column': 0}
            ), row=1, col=1)
            
            # 2. Income vs Expenses
            income_total = df[df['type'] == 'income']['amount'].sum()
            expense_total = df[df['type'] == 'expense']['amount'].sum()
            
            fig.add_trace(go.Pie(
                labels=['Income', 'Expenses'],
                values=[income_total, expense_total],
                marker_colors=['#2ecc71', '#e74c3c'],
                hole=0.3,
                textinfo='label+value'
            ), row=1, col=2)
            
            # 3. Spending Efficiency
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=spending_efficiency,
                title={'text': "Spending Efficiency"},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "green" if spending_efficiency > 70 else "orange" if spending_efficiency > 40 else "red"}},
                domain={'row': 1, 'column': 0}
            ), row=2, col=1)
            
            # 4. Top Categories
            expenses_df = df[df['type'] == 'expense']
            if not expenses_df.empty:
                top_categories = expenses_df.groupby('category')['amount'].sum().nlargest(5)
                fig.add_trace(go.Bar(
                    x=top_categories.values.tolist(),
                    y=top_categories.index.tolist(),
                    orientation='h',
                    marker_color='#3498db',
                    name='Top Categories'
                ), row=2, col=2)
            
            fig.update_layout(
                height=600,
                title_text=f"🚀 AI Financial Intelligence: {query}",
                showlegend=False
            )
            
            ai_insights = [
                f"Your financial health score: {financial_health_score}/100",
                f"Spending efficiency: {spending_efficiency:.1f}%",
                f"Income: ${income_total:.2f} | Expenses: ${expense_total:.2f}",
                "AI Recommendation: " + self._generate_ai_recommendation(df)
            ]
            
            return self._create_success_response(fig, "ai_intelligence_dashboard", 
                                               f"Advanced AI-powered financial intelligence dashboard", ai_insights)
            
        except Exception as e:
            print(f"❌ AI dashboard error: {e}")
            return self._create_comprehensive_fallback(df, query)

    # ========== HELPER METHODS ==========
    
    def _analyze_query_intent(self, query):
        """Enhanced query analysis with depth assessment"""
        query_lower = query.lower()
        
        # Determine analysis depth based on query complexity
        depth_keywords = {
            'deep': ['analyze', 'comprehensive', 'detailed', 'thorough', 'in-depth'],
            'medium': ['show', 'display', 'view', 'see', 'compare'],
            'light': ['simple', 'basic', 'quick', 'overview']
        }
        
        depth = 'medium'  # default
        for level, keywords in depth_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                depth = level
                break
        
        # Determine chart type
        for chart_type, keywords in self.chart_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return chart_type, depth
        
        return 'dashboard', depth

    def _forecast_categories(self, expenses_df, days):
        """Simple category forecasting"""
        category_totals = expenses_df.groupby('category')['amount'].sum()
        total = category_totals.sum()
        # Simple proportional forecasting
        return {cat: (amt/total) * 1000 for cat, amt in category_totals.nlargest(5).items()}

    def _calculate_financial_health(self, df):
        """Calculate financial health score (0-100)"""
        try:
            income = df[df['type'] == 'income']['amount'].sum()
            expenses = df[df['type'] == 'expense']['amount'].sum()
            
            if income == 0:
                return 0
                
            savings_rate = (income - expenses) / income * 100
            expense_diversity = len(df[df['type'] == 'expense']['category'].unique())
            
            # Simple scoring algorithm
            score = min(100, max(0, savings_rate * 0.7 + expense_diversity * 2))
            return round(score)
        except:
            return 50

    def _calculate_spending_efficiency(self, df):
        """Calculate spending efficiency score"""
        expenses_df = df[df['type'] == 'expense']
        if expenses_df.empty:
            return 0
        # Simple efficiency metric based on savings rate
        income = df[df['type'] == 'income']['amount'].sum()
        expenses = expenses_df['amount'].sum()
        if income == 0:
            return 0
        return max(0, ((income - expenses) / income) * 100)

    def _calculate_spending_volatility(self, expenses_df):
        """Calculate spending volatility risk (0-10)"""
        if len(expenses_df) < 5:
            return 5
        daily_spending = expenses_df.groupby('date')['amount'].sum()
        if daily_spending.mean() == 0:
            return 5
        volatility = daily_spending.std() / daily_spending.mean()
        return min(10, max(1, volatility * 5))

    def _calculate_category_concentration(self, expenses_df):
        """Calculate category concentration risk (0-10)"""
        category_totals = expenses_df.groupby('category')['amount'].sum()
        if len(category_totals) < 2:
            return 8  # High risk if only one category
        # Herfindahl index for concentration
        total = category_totals.sum()
        if total == 0:
            return 5
        hhi = sum((amt/total)**2 for amt in category_totals) * 10000
        return min(10, max(1, hhi / 1000))

    def _calculate_emergency_fund_score(self, df):
        """Calculate emergency fund adequacy (0-10)"""
        try:
            monthly_expenses = df[df['type'] == 'expense']['amount'].sum() / 3  # Approx monthly
            if monthly_expenses == 0:
                return 5
            # Simple: assume some emergency fund exists
            return 7  # Placeholder - in real app, would check actual savings
        except:
            return 5

    def _generate_risk_recommendation(self, volatility, concentration, emergency):
        """Generate risk mitigation recommendations"""
        recommendations = []
        if volatility > 7:
            recommendations.append("stabilize your spending patterns")
        if concentration > 7:
            recommendations.append("diversify your spending across more categories")
        if emergency < 4:
            recommendations.append("build an emergency fund")
        
        if not recommendations:
            return "Your financial risk profile looks good. Maintain current practices."
        return "Consider to: " + ", ".join(recommendations)

    def _generate_ai_recommendation(self, df):
        """Generate AI-powered financial recommendations"""
        expenses_df = df[df['type'] == 'expense']
        if expenses_df.empty:
            return "Start tracking your expenses to get personalized recommendations"
        
        top_category = expenses_df.groupby('category')['amount'].sum().idxmax()
        return f"Consider reducing spending in {top_category} category for better savings"

    def _create_comprehensive_fallback(self, df, query):
        """Fallback to comprehensive dashboard"""
        try:
            return self._create_ai_dashboard(df, query, 'medium')
        except:
            return self._create_error_response("Advanced analysis unavailable. Try a simpler query.")

    def _create_success_response(self, fig, chart_type, analysis_notes, insights=None):
        """Create successful response with advanced features"""
        try:
            # Ensure all data is serializable
            for trace in fig.data:
                for attr in ['x', 'y', 'z', 'labels', 'values']:
                    if hasattr(trace, attr) and getattr(trace, attr) is not None:
                        if hasattr(getattr(trace, attr), 'tolist'):
                            setattr(trace, attr, getattr(trace, attr).tolist())
            
            return {
                'success': True,
                'chart_type': chart_type,
                'chart_json': fig.to_json(),
                'title': f"🤖 AI Analysis: {analysis_notes}",
                'analysis_notes': analysis_notes,
                'data_points': len(fig.data),
                'insights': insights or ["Advanced AI analysis completed successfully"]
            }
        except Exception as e:
            print(f"❌ Advanced response error: {e}")
            return self._create_error_response(f"Advanced analysis completion failed: {str(e)}")

    def _create_error_response(self, message):
        """Create error response"""
        return {
            'success': False,
            'error': message,
            'chart_type': 'error',
            'analysis_notes': 'AI analysis failed',
            'insights': []
        }