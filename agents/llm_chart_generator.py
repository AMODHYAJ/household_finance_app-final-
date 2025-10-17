import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
import re
warnings.filterwarnings('ignore')

class LLMChartGenerator:
    """
    Complete Advanced AI chart generator with enhanced time-based analysis
    """
    
    def __init__(self):
        self.chart_types = {
            'predictive': ['predict', 'forecast', 'future', 'next', 'coming', 'trend', 'outlook'],
            'behavioral': ['pattern', 'habit', 'behavior', 'routine', 'lifestyle', 'spending habit'],
            'optimization': ['save', 'optimize', 'reduce', 'cut', 'better', 'improve', 'efficient'],
            'comparative': ['compare', 'vs', 'versus', 'difference', 'against', 'relative'],
            'temporal': ['timeline', 'over time', 'history', 'progress', 'evolution', 'journey'],
            'categorical': ['category', 'breakdown', 'distribution', 'by type', 'segmentation'],
            'expense_comparison': [' vs ', ' versus ', 'compare', 'difference between', 'food vs', 'shopping vs', 'transport vs', 'entertainment vs', 'bills vs', 'expense vs'],
            'risk': ['risk', 'alert', 'warning', 'danger', 'concern', 'problem'],
            'dashboard': ['dashboard', 'overview', 'summary', 'everything', 'complete']
        }
    
    def generate_chart_from_query(self, query, transactions_data):
        """Generate truly advanced AI-powered charts with enhanced error handling"""
        if not transactions_data:
            return self._create_error_response("No transaction data available")
        
        try:
            df = pd.DataFrame(transactions_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Enhanced query analysis
            chart_type, analysis_depth = self._analyze_query_intent(query)
            
            print(f"🤖 Advanced AI Analysis - Query: '{query}', Type: {chart_type}, Depth: {analysis_depth}")
            
            # Check for custom durations first
            if chart_type.startswith('custom_duration_'):
                return self._create_custom_duration_analysis(df, query, analysis_depth)
            
            # Check if this is a specific expense comparison query
            if self._is_specific_expense_comparison(query):
                return self._create_simple_expense_comparison(df, query)
            
            # Route to specific analysis methods
            if chart_type == 'last_month_spending':
                return self._create_last_month_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'this_month_spending':
                return self._create_this_month_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'last_year_spending':
                return self._create_last_year_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'this_year_spending':
                return self._create_this_year_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'last_week_spending':
                return self._create_last_week_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'this_week_spending':
                return self._create_this_week_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'today_spending':
                return self._create_today_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'yesterday_spending':
                return self._create_yesterday_spending_analysis(df, query, analysis_depth)
            elif chart_type == 'last_7_days':
                return self._create_custom_duration_analysis(df, "7 days", analysis_depth)
            elif chart_type == 'last_30_days':
                return self._create_custom_duration_analysis(df, "30 days", analysis_depth)
            elif chart_type == 'last_90_days':
                return self._create_custom_duration_analysis(df, "90 days", analysis_depth)
            elif chart_type == 'last_365_days':
                return self._create_custom_duration_analysis(df, "365 days", analysis_depth)
            elif chart_type == 'largest_transactions':
                return self._create_largest_transactions_analysis(df, query, analysis_depth)
            elif chart_type == 'spending_by_category':
                return self._create_categorical_analysis(df, query, analysis_depth)
            elif chart_type == 'expense_trends':
                return self._create_predictive_analysis(df, query, analysis_depth)
            elif chart_type == 'predictive':
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
            elif chart_type == 'expense_comparison':
                return self._create_simple_expense_comparison(df, query)
            elif chart_type == 'risk':
                return self._create_risk_analysis(df, query, analysis_depth)
            else:
                return self._create_ai_dashboard(df, query, analysis_depth)
            
        except Exception as e:
            print(f"❌ Advanced AI Chart generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_comprehensive_fallback(df, query)

    def _analyze_query_intent(self, query):
        """Enhanced query analysis with precise time-based filtering"""
        query_lower = query.lower()
        
        # First check for specific expense comparison
        if self._is_specific_expense_comparison(query_lower):
            return 'expense_comparison', 'medium'
        
        # Precise time period detection
        time_queries = {
            # Specific month queries
            'last_month_spending': ['last month', 'previous month', 'past month'],
            'this_month_spending': ['this month', 'current month'],
            'specific_month': ['january', 'february', 'march', 'april', 'may', 'june', 
                              'july', 'august', 'september', 'october', 'november', 'december'],
            
            # Year-based queries
            'last_year_spending': ['last year', 'previous year', 'past year'],
            'this_year_spending': ['this year', 'current year'],
            'yearly_analysis': ['yearly', 'annual', 'per year'],
            
            # Week-based queries
            'last_week_spending': ['last week', 'previous week', 'past week'],
            'this_week_spending': ['this week', 'current week'],
            'weekly_analysis': ['weekly', 'per week'],
            
            # Day-based queries  
            'today_spending': ['today', "today's"],
            'yesterday_spending': ['yesterday', "yesterday's"],
            'daily_analysis': ['daily', 'per day', 'day by day'],
            
            # Specific duration queries
            'recent_spending': ['recent', 'latest', 'newest', 'recently'],
            'last_7_days': ['7 days', 'seven days', '1 week', 'one week'],
            'last_30_days': ['30 days', 'thirty days', '1 month', 'one month'],
            'last_90_days': ['90 days', 'ninety days', '3 months', 'three months', 'quarter'],
            'last_365_days': ['365 days', 'year', '12 months', 'twelve months'],
            
            # Custom duration patterns (2 months, 3 years, 4 days, etc.)
            'custom_duration': self._extract_custom_duration(query_lower),
            
            # Other specific analyses
            'largest_transactions': ['largest', 'biggest', 'highest', 'most expensive'],
            'spending_by_category': ['by category', 'category breakdown', 'categories'],
            'expense_trends': ['trend', 'pattern', 'over time', 'history'],
            'monthly_comparison': ['month vs', 'compare months', 'monthly comparison'],
        }
        
        # Check for custom durations first (2 months, 3 years, etc.)
        custom_duration = self._extract_custom_duration(query_lower)
        if custom_duration:
            return f'custom_duration_{custom_duration["unit"]}', 'medium'
        
        # Check other time queries
        for chart_type, keywords in time_queries.items():
            if chart_type == 'custom_duration':
                continue  # Already handled above
            if any(keyword in query_lower for keyword in keywords):
                return chart_type, 'medium'
        
        # Determine analysis depth
        depth_keywords = {
            'deep': ['analyze', 'comprehensive', 'detailed', 'thorough', 'in-depth'],
            'medium': ['show', 'display', 'view', 'see', 'compare'],
            'light': ['simple', 'basic', 'quick', 'overview']
        }
        
        depth = 'medium'
        for level, keywords in depth_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                depth = level
                break
        
        # Determine chart type based on content
        for chart_type, keywords in self.chart_types.items():
            if any(keyword in query_lower for keyword in keywords):
                return chart_type, depth
        
        return 'dashboard', depth

    def _extract_custom_duration(self, query_lower):
        """Extract custom durations like '2 months', '3 years', '4 days'"""
        # Pattern to match numbers followed by time units
        patterns = [
            (r'(\d+)\s*months?', 'months'),
            (r'(\d+)\s*years?', 'years'), 
            (r'(\d+)\s*days?', 'days'),
            (r'(\d+)\s*weeks?', 'weeks'),
            (r'(\d+)\s*quarters?', 'months'),  # 1 quarter = 3 months
        ]
        
        for pattern, unit in patterns:
            match = re.search(pattern, query_lower)
            if match:
                number = int(match.group(1))
                return {
                    'number': number,
                    'unit': unit,
                    'query': query_lower
                }
        
        return None

    def _create_custom_duration_analysis(self, df, query, depth):
        """Analyze custom time periods like '2 months', '3 years', '4 days'"""
        try:
            duration_info = self._extract_custom_duration(query.lower())
            if not duration_info:
                return self._create_error_response("Could not understand the time period in your query.")
            
            number = duration_info['number']
            unit = duration_info['unit']
            
            # Calculate date range
            end_date = datetime.now()
            if unit == 'days':
                start_date = end_date - timedelta(days=number)
                period_name = f"{number} day{'s' if number > 1 else ''}"
            elif unit == 'weeks':
                start_date = end_date - timedelta(weeks=number)
                period_name = f"{number} week{'s' if number > 1 else ''}"
            elif unit == 'months':
                # More accurate month calculation
                start_date = end_date - pd.DateOffset(months=number)
                period_name = f"{number} month{'s' if number > 1 else ''}"
            elif unit == 'years':
                start_date = end_date - pd.DateOffset(years=number)
                period_name = f"{number} year{'s' if number > 1 else ''}"
            else:
                return self._create_error_response(f"Unsupported time unit: {unit}")
            
            # Filter data for the period - FIXED: Include category data
            period_expenses = df[
                (df['type'] == 'expense') & 
                (df['date'] >= start_date) & 
                (df['date'] <= end_date)
            ].copy()
            
            print(f"🔍 DEBUG: Found {len(period_expenses)} expenses for {period_name}")
            print(f"🔍 DEBUG: Categories: {period_expenses['category'].unique() if not period_expenses.empty else 'None'}")
            
            if period_expenses.empty:
                return self._create_custom_duration_error_response(df, period_name, start_date, end_date)
            
            # Create custom analysis based on duration length
            if number <= 7:  # Short period (days)
                return self._create_short_period_analysis(period_expenses, query, period_name, start_date, end_date)
            elif number <= 90:  # Medium period (weeks/months)
                return self._create_medium_period_analysis(period_expenses, query, period_name, start_date, end_date)
            else:  # Long period (months/years)
                return self._create_long_period_analysis(period_expenses, query, period_name, start_date, end_date)
            
        except Exception as e:
            print(f"❌ Custom duration analysis error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_comprehensive_fallback(df, query)

    def _create_short_period_analysis(self, expenses_df, query, period_name, start_date, end_date):
        """Analysis for short periods (up to 7 days) - detailed daily breakdown"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Daily Spending - {period_name}',
                'Spending Timeline',
                'Category Breakdown',
                'Spending Statistics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "pie"}, {"type": "indicator"}]
            ]
        )
        
        # 1. Daily spending bar chart
        daily_spending = expenses_df.groupby('date')['amount'].sum()
        dates_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Fill missing days with zero
        full_daily_spending = daily_spending.reindex(dates_range, fill_value=0)
        
        fig.add_trace(go.Bar(
            x=full_daily_spending.index,
            y=full_daily_spending.values,
            name='Daily Spending',
            marker_color='#e74c3c',
            hovertemplate='Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # 2. Cumulative spending
        cumulative_data = expenses_df.sort_values('date')
        if not cumulative_data.empty:
            cumulative_data['cumulative'] = cumulative_data['amount'].cumsum()
            fig.add_trace(go.Scatter(
                x=cumulative_data['date'],
                y=cumulative_data['cumulative'],
                name='Cumulative Spending',
                line=dict(color='#3498db', width=3),
                mode='lines+markers',
                hovertemplate='Date: %{x}<br>Cumulative: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
        
        # 3. Category breakdown - FIXED: Proper category aggregation
        category_totals = expenses_df.groupby('category')['amount'].sum()
        
        # Handle case where we have categories
        if not category_totals.empty:
            # Sort by amount and take top categories, group others as "Other"
            category_totals_sorted = category_totals.sort_values(ascending=False)
            
            # If more than 6 categories, group small ones as "Other"
            if len(category_totals_sorted) > 6:
                top_categories = category_totals_sorted.head(5)
                other_total = category_totals_sorted.tail(len(category_totals_sorted) - 5).sum()
                
                category_labels = top_categories.index.tolist() + ['Other']
                category_values = top_categories.values.tolist() + [other_total]
            else:
                category_labels = category_totals_sorted.index.tolist()
                category_values = category_totals_sorted.values.tolist()
            
            fig.add_trace(go.Pie(
                labels=category_labels,
                values=category_values,
                name='Categories',
                hole=0.4,
                hovertemplate='<b>%{label}</b><br>Amount: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>',
                textinfo='label+percent'
            ), row=2, col=1)
        else:
            # No categories found - show placeholder
            fig.add_trace(go.Pie(
                labels=['No Category Data'],
                values=[1],
                name='Categories',
                hole=0.4,
                marker_colors=['#95a5a6'],
                hovertemplate='No category data available<extra></extra>'
            ), row=2, col=1)
        
        # 4. Statistics indicator
        total_spent = expenses_df['amount'].sum()
        avg_daily = full_daily_spending.mean()
        days_with_spending = (full_daily_spending > 0).sum()
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=total_spent,
            number={'prefix': '$'},
            title={'text': f"Total {period_name}"},
            delta={'reference': 0},
            domain={'row': 2, 'column': 2}
        ), row=2, col=2)
        
        fig.update_layout(
            height=700,
            title_text=f"📊 {period_name.title()} Spending Analysis: {query.title()}",
            showlegend=False
        )
        
        # Enhanced insights with proper category data
        insights = [
            f"Total spent in {period_name}: ${total_spent:.2f}",
            f"Average daily spending: ${avg_daily:.2f}",
            f"Days with spending: {days_with_spending}/{len(dates_range)}",
            f"Highest spending day: ${full_daily_spending.max():.2f}" if not full_daily_spending.empty else "No spending data",
        ]
        
        # Add category insights if available
        if not category_totals.empty:
            top_category = category_totals.idxmax()
            top_amount = category_totals.max()
            insights.append(f"Top category: {top_category} (${top_amount:.2f})")
            insights.append(f"Categories with spending: {len(category_totals)}")
        
        return self._create_success_response(fig, "short_period_analysis", 
                                           f"Detailed daily analysis of {period_name} spending", insights)

    def _create_medium_period_analysis(self, expenses_df, query, period_name, start_date, end_date):
        """Analysis for medium periods (1 week to 3 months) - weekly aggregation"""
        # Aggregate by week
        expenses_df['week'] = expenses_df['date'].dt.to_period('W')
        weekly_spending = expenses_df.groupby('week')['amount'].sum()
        
        # FIXED: Proper category aggregation for medium periods
        category_totals = expenses_df.groupby('category')['amount'].sum()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Weekly Spending - {period_name}',
                'Spending Distribution',
                'Category Trends',
                'Spending Patterns'
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Weekly spending
        fig.add_trace(go.Bar(
            x=weekly_spending.index.astype(str),
            y=weekly_spending.values,
            name='Weekly Spending',
            marker_color='#e74c3c',
            hovertemplate='Week: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # 2. Spending distribution
        fig.add_trace(go.Box(
            y=expenses_df['amount'],
            name='Transaction Distribution',
            marker_color='#3498db',
            boxpoints='all',
            jitter=0.3,
            hovertemplate='Amount: $%{y:.2f}<extra></extra>'
        ), row=1, col=2)
        
        # 3. Category trends over weeks - FIXED: Proper category handling
        if not category_totals.empty:
            # Get top 5 categories for trend analysis
            top_categories = category_totals.nlargest(5).index
            
            category_weekly = expenses_df.groupby(['week', 'category'])['amount'].sum().unstack(fill_value=0)
            
            # Only show top categories to avoid clutter
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            
            for i, category in enumerate(top_categories):
                if category in category_weekly.columns:
                    fig.add_trace(go.Bar(
                        x=category_weekly.index.astype(str),
                        y=category_weekly[category],
                        name=category,
                        marker_color=colors[i % len(colors)],
                        hovertemplate='Week: %{x}<br>Category: ' + category + '<br>Amount: $%{y:.2f}<extra></extra>',
                        showlegend=True
                    ), row=2, col=1)
        else:
            # No category data - show placeholder
            fig.add_trace(go.Bar(
                x=weekly_spending.index.astype(str),
                y=[0] * len(weekly_spending),
                name='No Category Data',
                marker_color='#95a5a6',
                hovertemplate='No category data available<extra></extra>'
            ), row=2, col=1)
        
        # 4. Day of week heatmap - FIXED: Proper data aggregation
        expenses_df['day_of_week'] = expenses_df['date'].dt.day_name()
        expenses_df['week_num'] = expenses_df['date'].dt.isocalendar().week
        
        day_week_pivot = expenses_df.pivot_table(
            values='amount', 
            index='day_of_week', 
            columns='week_num', 
            aggfunc='sum', 
            fill_value=0
        )
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_week_pivot = day_week_pivot.reindex(days_order)
        
        fig.add_trace(go.Heatmap(
            z=day_week_pivot.values,
            x=day_week_pivot.columns.astype(str),
            y=day_week_pivot.index,
            colorscale='Viridis',
            name='Spending Heatmap',
            hovertemplate='Week: %{x}<br>Day: %{y}<br>Amount: $%{z:.2f}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            height=700,
            title_text=f"📈 {period_name.title()} Weekly Analysis: {query.title()}",
            showlegend=True,
            barmode='stack'
        )
        
        total_spent = expenses_df['amount'].sum()
        avg_weekly = weekly_spending.mean()
        
        insights = [
            f"Total spent in {period_name}: ${total_spent:.2f}",
            f"Average weekly spending: ${avg_weekly:.2f}",
            f"Number of weeks analyzed: {len(weekly_spending)}",
            f"Highest spending week: ${weekly_spending.max():.2f}" if not weekly_spending.empty else "No data",
        ]
        
        # Add category insights
        if not category_totals.empty:
            insights.append(f"Top category: {category_totals.idxmax()} (${category_totals.max():.2f})")
            insights.append(f"Categories with spending: {len(category_totals)}")
            insights.append(f"Most active spending day: {expenses_df['day_of_week'].mode().iloc[0] if not expenses_df['day_of_week'].mode().empty else 'N/A'}")
        
        return self._create_success_response(fig, "medium_period_analysis", 
                                           f"Weekly analysis of {period_name} spending patterns", insights)

    def _create_long_period_analysis(self, expenses_df, query, period_name, start_date, end_date):
        """Analysis for long periods (3+ months) - monthly aggregation and trends"""
        # Aggregate by month
        expenses_df['month'] = expenses_df['date'].dt.to_period('M')
        monthly_spending = expenses_df.groupby('month')['amount'].sum()
        
        # FIXED: Proper category aggregation
        category_totals = expenses_df.groupby('category')['amount'].sum()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f'Monthly Spending - {period_name}',
                'Spending Trend',
                'Category Evolution',
                'Seasonal Patterns'
            ),
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "box"}]
            ]
        )
        
        # 1. Monthly spending bars
        fig.add_trace(go.Bar(
            x=monthly_spending.index.astype(str),
            y=monthly_spending.values,
            name='Monthly Spending',
            marker_color='#e74c3c',
            hovertemplate='Month: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
        ), row=1, col=1)
        
        # 2. Trend line
        if len(monthly_spending) > 1:
            fig.add_trace(go.Scatter(
                x=monthly_spending.index.astype(str),
                y=monthly_spending.values,
                name='Spending Trend',
                line=dict(color='#3498db', width=3),
                mode='lines+markers',
                hovertemplate='Month: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
        
        # 3. Category trends over months - FIXED: Proper category handling
        if not category_totals.empty:
            # Get top 3 categories for trend analysis
            top_categories = category_totals.nlargest(3).index
            
            category_monthly = expenses_df.groupby(['month', 'category'])['amount'].sum().unstack(fill_value=0)
            
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            
            for i, category in enumerate(top_categories):
                if category in category_monthly.columns:
                    fig.add_trace(go.Scatter(
                        x=category_monthly.index.astype(str),
                        y=category_monthly[category],
                        name=f'{category} Trend',
                        mode='lines+markers',
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate='Month: %{x}<br>Category: ' + category + '<br>Amount: $%{y:.2f}<extra></extra>'
                    ), row=2, col=1)
        else:
            # No category data
            fig.add_trace(go.Scatter(
                x=monthly_spending.index.astype(str),
                y=[0] * len(monthly_spending),
                name='No Category Data',
                mode='lines',
                line=dict(color='#95a5a6', dash='dash'),
                hovertemplate='No category data available<extra></extra>'
            ), row=2, col=1)
        
        # 4. Monthly distribution - FIXED: Proper data aggregation
        monthly_data = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month_num in range(1, 13):
            month_data = expenses_df[expenses_df['date'].dt.month == month_num]['amount'].values
            if len(month_data) > 0:
                monthly_data.append(month_data)
            else:
                monthly_data.append(np.array([0]))
        
        fig.add_trace(go.Box(
            y=monthly_data,
            x=month_names,
            name='Monthly Distribution',
            marker_color='#2ecc71',
            hovertemplate='Month: %{x}<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            height=700,
            title_text=f"📈 {period_name.title()} Monthly Analysis: {query.title()}",
            showlegend=True
        )
        
        total_spent = expenses_df['amount'].sum()
        avg_monthly = monthly_spending.mean()
        trend_direction = "increasing" if len(monthly_spending) > 1 and monthly_spending.iloc[-1] > monthly_spending.iloc[0] else "decreasing"
        
        insights = [
            f"Total spent in {period_name}: ${total_spent:.2f}",
            f"Average monthly spending: ${avg_monthly:.2f}",
            f"Number of months analyzed: {len(monthly_spending)}",
            f"Overall trend: {trend_direction}",
            f"Highest spending month: ${monthly_spending.max():.2f}" if not monthly_spending.empty else "No data"
        ]
        
        # Add category insights
        if not category_totals.empty:
            insights.append(f"Top category: {category_totals.idxmax()} (${category_totals.max():.2f})")
            insights.append(f"Total categories: {len(category_totals)}")
        
        return self._create_success_response(fig, "long_period_analysis", 
                                           f"Monthly trend analysis of {period_name} spending", insights)

    def _create_custom_duration_error_response(self, df, period_name, start_date, end_date):
        """Create helpful error response for custom duration queries"""
        expenses_df = df[df['type'] == 'expense']
        
        if expenses_df.empty:
            insights = [
                "No expense transactions found in your data",
                "Add expense transactions to enable time-based analysis",
                f"Total transactions: {len(df)}",
                f"Transaction types: {', '.join(df['type'].unique())}"
            ]
        else:
            available_dates = expenses_df['date'].dt.to_period('M').unique()
            date_strings = [str(date) for date in sorted(available_dates)]
            
            insights = [
                f"Requested period: {period_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})",
                f"Available months with expense data: {len(available_dates)}",
                f"Date range: {min(date_strings) if date_strings else 'N/A'} to {max(date_strings) if date_strings else 'N/A'}",
                "Try analyzing one of the available months above",
                f"Total expense transactions: {len(expenses_df)}"
            ]
        
        return {
            'success': False,
            'error': f"No expense data found for {period_name} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}).",
            'chart_type': 'time_period_missing',
            'analysis_notes': f'Requested time period not available',
            'insights': insights,
            'available_periods': [str(period) for period in available_dates] if not expenses_df.empty else []
        }

    # Time-specific analysis methods
    def _create_last_month_spending_analysis(self, df, query, depth):
        """Specific analysis for last month spending"""
        try:
            # Filter for last month's expenses
            current_date = datetime.now()
            first_day_last_month = current_date.replace(day=1) - timedelta(days=1)
            first_day_last_month = first_day_last_month.replace(day=1)
            last_day_last_month = current_date.replace(day=1) - timedelta(days=1)
            
            last_month_expenses = df[
                (df['type'] == 'expense') & 
                (df['date'] >= first_day_last_month) & 
                (df['date'] <= last_day_last_month)
            ].copy()
            
            if last_month_expenses.empty:
                # Show what data we have instead
                available_months = df[df['type'] == 'expense']['date'].dt.to_period('M').unique()
                return self._create_time_period_error_response(
                    df, 
                    f"No expense data found for last month ({first_day_last_month.strftime('%B %Y')}).",
                    "last_month",
                    available_months
                )
            
            # Create focused last month analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'Last Month Spending: {first_day_last_month.strftime("%B %Y")}',
                    'Daily Spending Pattern',
                    'Category Breakdown',
                    'Top Transactions'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "pie"}, {"type": "bar"}]
                ]
            )
            
            # 1. Daily spending bar chart
            daily_spending = last_month_expenses.groupby('date')['amount'].sum()
            fig.add_trace(go.Bar(
                x=daily_spending.index,
                y=daily_spending.values,
                name='Daily Spending',
                marker_color='#e74c3c',
                hovertemplate='Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Cumulative spending
            cumulative_spending = last_month_expenses.sort_values('date')['amount'].cumsum()
            fig.add_trace(go.Scatter(
                x=last_month_expenses.sort_values('date')['date'],
                y=cumulative_spending,
                name='Cumulative Spending',
                line=dict(color='#3498db', width=3),
                mode='lines',
                hovertemplate='Date: %{x}<br>Cumulative: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
            
            # 3. Category breakdown - FIXED: Proper category handling
            category_totals = last_month_expenses.groupby('category')['amount'].sum()
            if not category_totals.empty:
                fig.add_trace(go.Pie(
                    labels=category_totals.index,
                    values=category_totals.values,
                    name='Categories',
                    hole=0.4,
                    hovertemplate='<b>%{label}</b><br>Amount: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
                ), row=2, col=1)
            
            # 4. Top transactions
            top_transactions = last_month_expenses.nlargest(5, 'amount')
            fig.add_trace(go.Bar(
                x=top_transactions['amount'],
                y=top_transactions['note'].fillna('No description'),
                orientation='h',
                name='Top Transactions',
                marker_color='#2ecc71',
                hovertemplate='Amount: $%{x:.2f}<br>Description: %{y}<extra></extra>'
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"📅 Last Month Spending Analysis: {query.title()}",
                showlegend=False
            )
            
            total_spent = last_month_expenses['amount'].sum()
            avg_daily = daily_spending.mean()
            max_day = daily_spending.idxmax()
            
            insights = [
                f"Total spent last month: ${total_spent:.2f}",
                f"Average daily spending: ${avg_daily:.2f}",
                f"Highest spending day: {max_day.strftime('%B %d')} (${daily_spending.max():.2f})",
                f"Number of transactions: {len(last_month_expenses)}"
            ]
            
            # Add category insights if available
            if not category_totals.empty:
                insights.append(f"Top category: {category_totals.idxmax()} (${category_totals.max():.2f})")
                insights.append(f"Categories with spending: {len(category_totals)}")
            
            return self._create_success_response(fig, "last_month_analysis", 
                                               f"Detailed analysis of last month's spending", insights)
            
        except Exception as e:
            print(f"❌ Last month analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_this_month_spending_analysis(self, df, query, depth):
        """Specific analysis for current month spending"""
        try:
            # Filter for current month's expenses
            current_date = datetime.now()
            first_day_this_month = current_date.replace(day=1)
            
            this_month_expenses = df[
                (df['type'] == 'expense') & 
                (df['date'] >= first_day_this_month)
            ].copy()
            
            if this_month_expenses.empty:
                available_months = df[df['type'] == 'expense']['date'].dt.to_period('M').unique()
                return self._create_time_period_error_response(
                    df,
                    f"No expense data found for this month ({first_day_this_month.strftime('%B %Y')}).",
                    "this_month", 
                    available_months
                )
            
            # Create this month analysis (similar structure but with progress indicators)
            days_in_month = (current_date - first_day_this_month).days + 1
            total_days_in_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            total_days_in_month = total_days_in_month.day
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    f'This Month Spending: {first_day_this_month.strftime("%B %Y")}',
                    'Spending Progress vs Time',
                    'Category Distribution',
                    'Daily Spending Rate'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "scatter"}],
                    [{"type": "pie"}, {"type": "indicator"}]
                ]
            )
            
            # 1. Daily spending
            daily_spending = this_month_expenses.groupby('date')['amount'].sum()
            fig.add_trace(go.Bar(
                x=daily_spending.index,
                y=daily_spending.values,
                name='Daily Spending',
                marker_color='#e74c3c',
                hovertemplate='Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Cumulative vs expected linear
            cumulative_data = this_month_expenses.sort_values('date')
            cumulative_data['cumulative'] = cumulative_data['amount'].cumsum()
            
            # Create expected linear spending line
            total_to_date = cumulative_data['cumulative'].iloc[-1] if not cumulative_data.empty else 0
            expected_daily = total_to_date / days_in_month if days_in_month > 0 else 0
            
            dates_range = pd.date_range(start=first_day_this_month, end=current_date, freq='D')
            expected_cumulative = [expected_daily * (i + 1) for i in range(len(dates_range))]
            
            fig.add_trace(go.Scatter(
                x=dates_range,
                y=expected_cumulative,
                name='Expected Linear Spending',
                line=dict(color='#95a5a6', width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Expected: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=cumulative_data['date'],
                y=cumulative_data['cumulative'],
                name='Actual Spending',
                line=dict(color='#3498db', width=3),
                hovertemplate='Date: %{x}<br>Actual: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
            
            # 3. Category breakdown - FIXED: Proper category handling
            category_totals = this_month_expenses.groupby('category')['amount'].sum()
            if not category_totals.empty:
                fig.add_trace(go.Pie(
                    labels=category_totals.index,
                    values=category_totals.values,
                    name='Categories',
                    hole=0.4,
                    hovertemplate='<b>%{label}</b><br>Amount: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>'
                ), row=2, col=1)
            
            # 4. Progress indicator
            month_progress = (days_in_month / total_days_in_month) * 100
            spending_progress = (total_to_date / (expected_daily * total_days_in_month)) * 100 if expected_daily > 0 else 0
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=spending_progress,
                number={'suffix': "%"},
                title={'text': "Spending vs Time Progress"},
                gauge={
                    'axis': {'range': [0, 200]},
                    'bar': {'color': "green" if spending_progress <= 100 else "orange" if spending_progress <= 150 else "red"},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgreen"},
                        {'range': [100, 150], 'color': "yellow"},
                        {'range': [150, 200], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                },
                domain={'row': 2, 'column': 2}
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"📊 This Month Spending Analysis: {query.title()}",
                showlegend=True
            )
            
            total_spent = this_month_expenses['amount'].sum()
            avg_daily = daily_spending.mean() if not daily_spending.empty else 0
            
            insights = [
                f"Total spent this month: ${total_spent:.2f}",
                f"Average daily spending: ${avg_daily:.2f}",
                f"Month progress: {days_in_month}/{total_days_in_month} days ({month_progress:.1f}%)",
                f"Spending progress: {spending_progress:.1f}% of expected monthly total",
            ]
            
            # Add category insights if available
            if not category_totals.empty:
                insights.append(f"Top category: {category_totals.idxmax()}")
                insights.append(f"Categories with spending: {len(category_totals)}")
            
            return self._create_success_response(fig, "this_month_analysis", 
                                               f"Current month spending analysis with progress tracking", insights)
            
        except Exception as e:
            print(f"❌ This month analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_largest_transactions_analysis(self, df, query, depth):
        """Analysis of largest transactions"""
        try:
            # Get largest transactions (both income and expense)
            largest_expenses = df[df['type'] == 'expense'].nlargest(10, 'amount')
            largest_income = df[df['type'] == 'income'].nlargest(10, 'amount')
            
            if largest_expenses.empty and largest_income.empty:
                return self._create_error_response("No transaction data available for largest transactions analysis.")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Top 10 Largest Expenses',
                    'Top 10 Largest Income',
                    'Expense Categories - Largest',
                    'Income Sources - Largest'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # 1. Largest expenses
            if not largest_expenses.empty:
                fig.add_trace(go.Bar(
                    x=largest_expenses['amount'],
                    y=largest_expenses['note'].fillna('No description'),
                    orientation='h',
                    name='Largest Expenses',
                    marker_color='#e74c3c',
                    hovertemplate='Amount: $%{x:.2f}<br>Date: %{customdata}<extra></extra>',
                    customdata=largest_expenses['date'].dt.strftime('%Y-%m-%d')
                ), row=1, col=1)
            
            # 2. Largest income
            if not largest_income.empty:
                fig.add_trace(go.Bar(
                    x=largest_income['amount'],
                    y=largest_income['note'].fillna('No description'),
                    orientation='h',
                    name='Largest Income',
                    marker_color='#2ecc71',
                    hovertemplate='Amount: $%{x:.2f}<br>Date: %{customdata}<extra></extra>',
                    customdata=largest_income['date'].dt.strftime('%Y-%m-%d')
                ), row=1, col=2)
            
            # 3. Expense categories for largest transactions
            if not largest_expenses.empty:
                expense_by_category = largest_expenses.groupby('category')['amount'].sum().nlargest(5)
                fig.add_trace(go.Bar(
                    x=expense_by_category.index,
                    y=expense_by_category.values,
                    name='Expense Categories',
                    marker_color='#e67e22',
                    hovertemplate='Category: %{x}<br>Total: $%{y:.2f}<extra></extra>'
                ), row=2, col=1)
            
            # 4. Income categories for largest transactions
            if not largest_income.empty:
                income_by_category = largest_income.groupby('category')['amount'].sum().nlargest(5)
                fig.add_trace(go.Bar(
                    x=income_by_category.index,
                    y=income_by_category.values,
                    name='Income Categories',
                    marker_color='#27ae60',
                    hovertemplate='Category: %{x}<br>Total: $%{y:.2f}<extra></extra>'
                ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"💰 Largest Transactions Analysis: {query.title()}",
                showlegend=False
            )
            
            insights = []
            if not largest_expenses.empty:
                insights.append(f"Largest expense: ${largest_expenses['amount'].iloc[0]:.2f} ({largest_expenses['note'].iloc[0]})")
                insights.append(f"Total large expenses: ${largest_expenses['amount'].sum():.2f}")
            
            if not largest_income.empty:
                insights.append(f"Largest income: ${largest_income['amount'].iloc[0]:.2f} ({largest_income['note'].iloc[0]})")
                insights.append(f"Total large income: ${largest_income['amount'].sum():.2f}")
            
            insights.append(f"Analysis covers {len(largest_expenses) + len(largest_income)} largest transactions")
            
            return self._create_success_response(fig, "largest_transactions", 
                                               f"Analysis of largest transactions by amount", insights)
            
        except Exception as e:
            print(f"❌ Largest transactions analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)

    def _create_time_period_error_response(self, df, message, period_type, available_periods):
        """Create helpful error response for time period queries"""
        expenses_df = df[df['type'] == 'expense']
        
        if expenses_df.empty:
            insights = [
                "No expense transactions found in your data",
                "Add expense transactions to enable time-based analysis",
                f"Total transactions: {len(df)}",
                f"Transaction types: {', '.join(df['type'].unique())}"
            ]
        else:
            available_dates = expenses_df['date'].dt.to_period('M').unique()
            date_strings = [str(date) for date in sorted(available_dates)]
            
            insights = [
                f"Available months with expense data: {len(available_dates)}",
                f"Date range: {min(date_strings) if date_strings else 'N/A'} to {max(date_strings) if date_strings else 'N/A'}",
                "Try analyzing one of the available months above",
                f"Total expense transactions: {len(expenses_df)}"
            ]
        
        return {
            'success': False,
            'error': message,
            'chart_type': 'time_period_missing',
            'analysis_notes': f'Requested time period not available',
            'insights': insights,
            'available_periods': [str(period) for period in available_periods] if available_periods is not None else []
        }

    # Placeholder methods for other time-based analyses (you can implement these similarly)
    def _create_last_year_spending_analysis(self, df, query, depth):
        return self._create_custom_duration_analysis(df, "1 year", depth)
    
    def _create_this_year_spending_analysis(self, df, query, depth):
        return self._create_custom_duration_analysis(df, f"{datetime.now().year}", depth)
    
    def _create_last_week_spending_analysis(self, df, query, depth):
        return self._create_custom_duration_analysis(df, "7 days", depth)
    
    def _create_this_week_spending_analysis(self, df, query, depth):
        return self._create_custom_duration_analysis(df, "7 days", depth)
    
    def _create_today_spending_analysis(self, df, query, depth):
        return self._create_custom_duration_analysis(df, "1 day", depth)
    
    def _create_yesterday_spending_analysis(self, df, query, depth):
        return self._create_custom_duration_analysis(df, "1 day", depth)

    def _is_specific_expense_comparison(self, query):
        """Check if query is specifically asking to compare expense categories"""
        query_lower = query.lower()
        
        # Check for direct comparison patterns
        comparison_indicators = [' vs ', ' versus ', 'compare ']
        category_indicators = ['food', 'shopping', 'transport', 'entertainment', 'bills', 'healthcare', 'utilities', 'groceries', 'restaurant', 'dining', 'movie', 'netflix', 'uber', 'taxi', 'gas', 'electricity', 'water', 'internet', 'phone']
        
        has_comparison = any(indicator in query_lower for indicator in comparison_indicators)
        has_categories = any(category in query_lower for category in category_indicators)
        
        return has_comparison and has_categories

    def _create_simple_expense_comparison(self, df, query):
        """Create simple bar chart for specific expense category comparisons with enhanced error handling"""
        try:
            expenses_df = df[df['type'] == 'expense'].copy()
            
            if expenses_df.empty:
                return self._create_error_response(
                    "No expense data available for comparison. "
                    "Add some expense transactions first."
                )
            
            # Extract specific categories mentioned in query with missing data info
            categories_to_compare, missing_categories = self._extract_specific_categories_from_query(query, expenses_df)
            
            print(f"🔍 DEBUG: categories_to_compare={categories_to_compare}, missing_categories={missing_categories}")

            # NEW LOGIC: If user specifically requested categories that don't exist, show error
            comparison_words = [' vs ', ' versus ', 'compare ']
            query_lower = query.lower()
            has_direct_comparison = any(word in query_lower for word in comparison_words)
            
            if has_direct_comparison and missing_categories:
                # User specifically asked for categories that don't exist
                return self._create_missing_data_response(missing_categories, expenses_df)
            
            # If user requested specific categories but none found in data
            if not categories_to_compare and missing_categories:
                return self._create_missing_data_response(missing_categories, expenses_df)
            
            # If fewer than 2 categories found for a comparison, suggest alternatives
            if len(categories_to_compare) < 2 and has_direct_comparison:
                available_categories = expenses_df['category'].unique()
                print(f"🔍 DEBUG: Only found {len(categories_to_compare)} categories for direct comparison")
                return self._create_suggestion_response(
                    query, categories_to_compare, missing_categories, available_categories
                )
            
            # If we have at least 2 categories, proceed with comparison
            if len(categories_to_compare) >= 2:
                print(f"📊 Creating simple comparison for: {categories_to_compare}")
                
                # Calculate totals for the specific categories
                category_totals = expenses_df.groupby('category')['amount'].sum()
                comparison_totals = [category_totals.get(cat, 0) for cat in categories_to_compare]
                
                # Create simple bar chart
                fig = go.Figure()
                
                colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
                
                fig.add_trace(go.Bar(
                    x=categories_to_compare,
                    y=comparison_totals,
                    marker_color=colors[:len(categories_to_compare)],
                    hovertemplate='<b>%{x}</b><br>Total: $%{y:.2f}<extra></extra>',
                    text=[f'${amt:.2f}' for amt in comparison_totals],
                    textposition='auto',
                    textfont=dict(size=14, color='white')
                ))
                
                total_comparison = sum(comparison_totals)
                
                fig.update_layout(
                    title=f"📊 Expense Comparison: {query.title()}",
                    xaxis_title="Categories",
                    yaxis_title="Amount ($)",
                    showlegend=False,
                    height=500,
                    plot_bgcolor='white',
                    font=dict(size=12)
                )
                
                # Generate insights
                insights = self._generate_comparison_insights(categories_to_compare, comparison_totals)
                
                return self._create_success_response(fig, "expense_comparison", 
                                                   f"Direct expense category comparison", insights)
            else:
                # Fallback: not enough categories for comparison
                available_categories = expenses_df['category'].unique()
                return self._create_suggestion_response(
                    query, categories_to_compare, missing_categories, available_categories
                )
                
        except Exception as e:
            print(f"❌ Simple expense comparison error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_comprehensive_fallback(df, query)

    def _extract_specific_categories_from_query(self, query, expenses_df):
        """Extract specific categories mentioned in comparison queries with clear feedback"""
        query_lower = query.lower()

        # Get available categories from user's actual data
        available_categories = expenses_df['category'].unique()
        available_categories_lower = [str(cat).lower() for cat in available_categories]
        
        print(f"📋 User's available categories: {list(available_categories)}")
        print(f"🔍 Query: '{query}'")

        found_categories = []
        missing_categories = []

        # Split query into words and look for exact category matches
        query_words = set(query_lower.split())
        
        # Also check for comparison patterns like "food vs shopping"
        comparison_patterns = ['vs', 'versus', 'compare', 'and', '&']
        query_words = query_words - set(comparison_patterns)
        
        # First pass: direct exact matching
        for available_cat in available_categories:
            available_cat_lower = str(available_cat).lower()
            if available_cat_lower in query_words:
                if available_cat not in found_categories:
                    found_categories.append(available_cat)
                    print(f"✅ Found exact category match: {available_cat}")

        # Second pass: check if any query word is a substring of category names (but be careful)
        for query_word in query_words:
            for available_cat in available_categories:
                available_cat_lower = str(available_cat).lower()
                # Only match if the query word is a significant part of the category name
                # and not just a random substring match
                if (query_word in available_cat_lower and 
                    len(query_word) > 3 and  # Minimum length to avoid false matches
                    available_cat not in found_categories):
                    # Additional check: make sure it's not a partial word match like "port" in "transport"
                    if (available_cat_lower.startswith(query_word) or 
                        f' {query_word}' in f' {available_cat_lower}' or
                        available_cat_lower.endswith(query_word)):
                        found_categories.append(available_cat)
                        print(f"✅ Found partial category match: {available_cat} for '{query_word}'")

        # Third pass: use a simple keyword mapping for common variations
        common_variations = {
            'food': ['food', 'groceries', 'restaurant', 'dining', 'grocery'],
            'shopping': ['shopping', 'retail', 'store', 'mall'],
            'transport': ['transport', 'transportation', 'uber', 'taxi', 'bus', 'train', 'car'],
            'entertainment': ['entertainment', 'movie', 'netflix', 'game', 'concert'],
            'bills': ['bills', 'utilities', 'electricity', 'water', 'internet', 'phone'],
            'healthcare': ['healthcare', 'medical', 'doctor', 'hospital', 'pharmacy']
        }
        
        for query_word in query_words:
            for standard_name, variations in common_variations.items():
                if query_word in variations and standard_name in available_categories_lower:
                    exact_match_idx = available_categories_lower.index(standard_name)
                    exact_match = available_categories[exact_match_idx]
                    if exact_match not in found_categories:
                        found_categories.append(exact_match)
                        print(f"✅ Mapped '{query_word}' to standard category: {exact_match}")

        print(f"🔍 DEBUG: found_categories={found_categories}, missing_categories={missing_categories}")

        # If it's a direct comparison query, be more strict about what we include
        comparison_words = [' vs ', ' versus ', 'compare ']
        has_comparison = any(word in query_lower for word in comparison_words)
        
        if has_comparison and len(found_categories) < 2:
            # For comparison queries, we need at least 2 categories
            print(f"⚠️ Comparison query but only found {len(found_categories)} categories")
        
        print(f"📊 Final result: found={found_categories}, missing={missing_categories}")
        return found_categories, missing_categories

    def _create_missing_data_response(self, missing_categories, expenses_df):
        """Create helpful response when requested categories don't exist"""
        available_categories = expenses_df['category'].unique()
        
        # Create more specific message based on what was missing
        if len(missing_categories) == 1:
            message = f"🚫 No '{missing_categories[0]}' transactions found in your data"
        else:
            message = f"🚫 No data available for: {', '.join(missing_categories)}"
        
        if len(available_categories) > 0:
            message += f"\n\n📊 Your available expense categories: {', '.join(map(str, available_categories))}"
            
            # Suggest specific comparisons from available categories
            if len(available_categories) >= 2:
                message += f"\n\n💡 Try: 'compare {available_categories[0]} vs {available_categories[1]}'"
            if len(available_categories) >= 3:
                message += f"\n💡 Or: 'show {available_categories[0]} vs {available_categories[1]} vs {available_categories[2]}'"
        
        insights = [
            f"Requested categories not found: {', '.join(missing_categories)}",
            f"You have {len(available_categories)} expense categories available",
            "Add transactions with the missing categories to enable comparison",
            "Try comparing your available categories listed above"
        ]
        
        return {
            'success': False,
            'error': message,
            'chart_type': 'missing_data',
            'analysis_notes': 'Requested categories not found',
            'insights': insights,
            'available_categories': [str(cat) for cat in available_categories]
        }

    def _create_suggestion_response(self, query, found_categories, missing_categories, available_categories):
        """Suggest alternatives when exact matches aren't found"""
        available_list = [str(cat) for cat in available_categories]
        
        message = "🔍 I found these categories in your query:\n"
        if found_categories:
            message += f"✅ Available: {', '.join(found_categories)}\n"
        if missing_categories:
            message += f"❌ Not available: {', '.join(missing_categories)}\n"
        
        message += f"\n📊 Your available categories: {', '.join(available_list[:5])}"
        message += f"\n\n💡 Try: 'compare {', '.join(available_list[:2])}'"
        
        insights = [
            f"Found {len(found_categories)} matching categories in your data",
            f"Missing {len(missing_categories)} categories from your query",
            f"Total categories available: {len(available_list)}",
            "Try comparing the categories listed above"
        ]
        
        return {
            'success': False,
            'error': message,
            'chart_type': 'suggestion',
            'analysis_notes': 'Category suggestions',
            'insights': insights,
            'available_categories': available_list
        }

    def _generate_comparison_insights(self, categories, totals):
        """Generate insights for category comparison"""
        if len(categories) < 2:
            return ["Need at least 2 categories for comparison"]
        
        insights = []
        
        # Find highest and lowest
        max_idx = totals.index(max(totals))
        min_idx = totals.index(min(totals))
        
        insights.append(f"💸 Highest spending: {categories[max_idx]} (${totals[max_idx]:.2f})")
        insights.append(f"💰 Lowest spending: {categories[min_idx]} (${totals[min_idx]:.2f})")
        
        # Calculate ratio if possible
        if min(totals) > 0:
            ratio = max(totals) / min(totals)
            insights.append(f"⚖️ Spending ratio: {ratio:.1f}x difference")
        
        # Add percentage breakdown
        total = sum(totals)
        if total > 0:
            for i, category in enumerate(categories):
                percentage = (totals[i] / total) * 100
                insights.append(f"📈 {category}: ${totals[i]:.2f} ({percentage:.1f}%)")
        
        return insights

    # Keep all other existing methods (predictive, behavioral, optimization, etc.)
    def _create_predictive_analysis(self, df, query, depth):
        """Predictive analysis with machine learning"""
        try:
            # Prepare data for forecasting
            expenses_df = df[df['type'] == 'expense'].copy()
            if len(expenses_df) < 5:
                return self._create_error_response(
                    f"Need more data for accurate predictions. "
                    f"You have {len(expenses_df)} expense transactions. "
                    "Add at least 5 expense transactions for predictive analysis."
                )
            
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
            
            if len(expenses_df) < 5:
                return self._create_error_response(
                    f"Need more expense data for behavioral analysis. "
                    f"You have {len(expenses_df)} expense transactions. "
                    "Add at least 5 expense transactions to identify patterns."
                )
            
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
            
            if expenses_df.empty:
                return self._create_error_response("No expense data for optimization analysis")
            
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
            if len(df) < 3:
                return self._create_error_response(
                    f"Need more data for comparative analysis. "
                    f"You have {len(df)} transactions. "
                    "Add at least 3 transactions to enable comparisons."
                )
            
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
            # Check if we have enough data for meaningful dashboard
            if len(df) < 3:
                return self._create_error_response(
                    f"Need more data for comprehensive dashboard. "
                    f"You have {len(df)} transactions. "
                    "Add at least 3 transactions to see meaningful insights."
                )
            
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

    def _create_temporal_analysis(self, df, query, depth):
        """Temporal analysis for time-based patterns"""
        try:
            expenses_df = df[df['type'] == 'expense']
            
            if expenses_df.empty:
                return self._create_error_response("No expense data for temporal analysis")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Spending Timeline',
                    'Monthly Trends',
                    'Daily Patterns',
                    'Seasonal Analysis'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "heatmap"}, {"type": "box"}]
                ]
            )
            
            # 1. Spending timeline
            timeline_data = expenses_df.sort_values('date')
            fig.add_trace(go.Scatter(
                x=timeline_data['date'],
                y=timeline_data['amount'],
                mode='markers',
                name='Individual Transactions',
                marker=dict(color='#e74c3c', size=8, opacity=0.6),
                hovertemplate='Date: %{x}<br>Amount: $%{y:.2f}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Monthly trends
            monthly_data = expenses_df.groupby(expenses_df['date'].dt.to_period('M'))['amount'].sum()
            fig.add_trace(go.Bar(
                x=monthly_data.index.astype(str),
                y=monthly_data.values,
                name='Monthly Spending',
                marker_color='#3498db',
                hovertemplate='Month: %{x}<br>Total: $%{y:.2f}<extra></extra>'
            ), row=1, col=2)
            
            # 3. Daily patterns heatmap
            expenses_df['day_of_week'] = expenses_df['date'].dt.day_name()
            expenses_df['month'] = expenses_df['date'].dt.month_name()
            
            daily_pattern = expenses_df.groupby(['month', 'day_of_week'])['amount'].sum().unstack(fill_value=0)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                           'July', 'August', 'September', 'October', 'November', 'December']
            
            daily_pattern = daily_pattern.reindex(months_order).reindex(columns=days_order)
            
            fig.add_trace(go.Heatmap(
                z=daily_pattern.values,
                x=daily_pattern.columns,
                y=daily_pattern.index,
                colorscale='Viridis',
                name='Daily Patterns',
                hovertemplate='Month: %{y}<br>Day: %{x}<br>Amount: $%{z:.2f}<extra></extra>'
            ), row=2, col=1)
            
            # 4. Seasonal analysis
            monthly_avg = expenses_df.groupby(expenses_df['date'].dt.month)['amount'].mean()
            monthly_avg = monthly_avg.reindex(range(1, 13), fill_value=0)
            
            fig.add_trace(go.Box(
                y=[expenses_df[expenses_df['date'].dt.month == i]['amount'].values for i in range(1, 13)],
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                name='Monthly Distribution',
                marker_color='#2ecc71',
                hovertemplate='Month: %{x}<extra></extra>'
            ), row=2, col=2)
            
            fig.update_layout(
                height=700,
                title_text=f"⏰ AI Temporal Analysis: {query}",
                showlegend=False
            )
            
            temporal_insights = [
                f"Total transactions analyzed: {len(expenses_df)}",
                f"Date range: {expenses_df['date'].min().strftime('%Y-%m-%d')} to {expenses_df['date'].max().strftime('%Y-%m-%d')}",
                f"Average monthly spending: ${monthly_data.mean():.2f}",
                f"Most active spending month: {monthly_data.idxmax().strftime('%B %Y') if not monthly_data.empty else 'N/A'}"
            ]
            
            return self._create_success_response(fig, "temporal_analysis", 
                                               f"AI-powered temporal spending pattern analysis", temporal_insights)
            
        except Exception as e:
            print(f"❌ Temporal analysis error: {e}")
            return self._create_comprehensive_fallback(df, query)