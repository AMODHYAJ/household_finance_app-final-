import re
import google.generativeai as genai
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta

class LLMChartGenerator:
    """LLM-powered chart generation from natural language requests"""
    
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.available = False
        
        if self.api_key and not self.api_key.startswith('your-'):
            try:
                genai.configure(api_key=self.api_key)
                # Try different model names
                try:
                    self.model = genai.GenerativeModel('gemini-2.0-flash')
                    self.available = True
                except:
                    try:
                        self.model = genai.GenerativeModel('gemini-pro')
                        self.available = True
                    except:
                        try:
                            self.model = genai.GenerativeModel('gemini-1.5-flash')
                            self.available = True
                        except Exception as e:
                            print(f"❌ No Gemini model available: {e}")
                            self.available = False
                
                if self.available:
                    print("✅ LLM Chart Generator loaded with Gemini")
            except Exception as e:
                print(f"❌ LLM Chart Generator error: {e}")
                self.available = False
        else:
            print("⚠️ Gemini API key not configured for chart generation")
            self.available = False
    
    def generate_chart_from_query(self, query: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Generate chart based on natural language query"""
        # First check for specific query patterns that need special handling
        special_result = self._handle_special_queries(query, transactions)
        if special_result:
            return special_result
        
        if not self.available:
            return self._fallback_chart_generation(query, transactions)
        
        try:
            # Prepare transaction summary for LLM
            summary = self._prepare_chart_data_summary(transactions)
            
            prompt = f"""
            Analyze this financial data and user request, then generate an appropriate chart configuration.
            
            FINANCIAL DATA SUMMARY:
            {summary}
            
            USER REQUEST: "{query}"
            
            IMPORTANT SPECIAL CASES:
            - If query asks for "largest transactions", "biggest transactions", or "top transactions", create a bar chart of individual transactions sorted by amount
            - If query asks for "income vs expenses over time", create a line chart showing monthly trends
            - If query compares categories (uses "vs", "versus", "and", "compare"), create a bar chart comparing ONLY those categories
            - For category distributions, use pie charts
            - For trends over time, use line charts
            - For individual transaction analysis, use bar charts
            
            RESPONSE FORMAT (JSON only):
            {{
                "chart_type": "pie|bar|line|scatter",
                "title": "Chart title based on user request",
                "x_axis": "field_name_for_x_axis",
                "y_axis": "field_name_for_y_axis", 
                "color_by": "field_name_for_colors",
                "filters": {{
                    "time_period": "last_month|last_3_months|all_time",
                    "categories": ["category1", "category2"],
                    "min_amount": 0,
                    "max_amount": null
                }},
                "analysis_notes": "Brief explanation of what this chart shows",
                "insights": ["key insight 1", "key insight 2"]
            }}
            
            Available fields: date, type, category, amount, note
            Chart types: pie (for categories), bar (comparisons), line (trends), scatter (relationships)
            
            Focus on creating useful, actionable financial visualizations.
            """
            
            response = self.model.generate_content(prompt)
            chart_config = self._parse_llm_response(response.text)
            
            # Generate the actual chart
            chart_result = self._create_chart_from_config(chart_config, transactions)
            
            return chart_result
            
        except Exception as e:
            logging.error(f"LLM chart generation failed: {e}")
            return self._fallback_chart_generation(query, transactions)
    
    def _handle_special_queries(self, query: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Handle specific query patterns that need special handling"""
        query_lower = query.lower()
        
        # Handle "largest transactions" queries
        if any(word in query_lower for word in ['largest', 'biggest', 'top transactions', 'highest']):
            return self._create_largest_transactions_chart(transactions, query)
        
        # Handle "income vs expenses over time" queries
        if any(word in query_lower for word in ['income vs expenses over time', 'income and expenses over time', 'revenue vs spending over time']):
            return self._create_income_expense_trend_chart(transactions, query)
        
        # Handle category comparisons
        comparison_result = self._handle_comparison_query(query, transactions)
        if comparison_result:
            return comparison_result
        
        return None
    
    def _handle_comparison_query(self, query: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Handle comparison queries like 'entertainment vs food'"""
        query_lower = query.lower()
        
        # Look for comparison patterns
        vs_match = re.search(r'(\w+)\s+vs\s+(\w+)', query_lower)
        if vs_match:
            category1 = vs_match.group(1).title()
            category2 = vs_match.group(2).title()
            return self._create_category_comparison_chart(transactions, [category1, category2], query)
        
        and_match = re.search(r'(\w+)\s+and\s+(\w+)', query_lower)
        if and_match:
            category1 = and_match.group(1).title()
            category2 = and_match.group(2).title()
            return self._create_category_comparison_chart(transactions, [category1, category2], query)
        
        compare_match = re.search(r'compare\s+(\w+)\s+and\s+(\w+)', query_lower)
        if compare_match:
            category1 = compare_match.group(1).title()
            category2 = compare_match.group(2).title()
            return self._create_category_comparison_chart(transactions, [category1, category2], query)
        
        versus_match = re.search(r'(\w+)\s+versus\s+(\w+)', query_lower)
        if versus_match:
            category1 = versus_match.group(1).title()
            category2 = versus_match.group(2).title()
            return self._create_category_comparison_chart(transactions, [category1, category2], query)
        
        return None
    
    def _create_largest_transactions_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create a chart showing the largest individual transactions"""
        df = pd.DataFrame(transactions)
        
        if df.empty:
            return self._create_empty_chart("No transaction data available")
        
        # Filter for this month if specified
        query_lower = query.lower()
        if 'month' in query_lower:
            current_month = datetime.now().strftime('%Y-%m')
            df['date'] = pd.to_datetime(df['date'])
            filtered_df = df[df['date'].dt.strftime('%Y-%m') == current_month]
            time_period = "this month"
        elif 'week' in query_lower:
            current_week = datetime.now().strftime('%Y-%U')
            df['date'] = pd.to_datetime(df['date'])
            filtered_df = df[df['date'].dt.strftime('%Y-%U') == current_week]
            time_period = "this week"
        else:
            filtered_df = df
            time_period = "all time"
        
        if filtered_df.empty:
            return self._create_empty_chart(f"No transactions found for {time_period}")
        
        # Get top 10 largest transactions (by absolute amount)
        largest_transactions = filtered_df.nlargest(10, 'amount')
        
        if largest_transactions.empty:
            return self._create_empty_chart("No transactions found")
        
        # Create horizontal bar chart for better readability
        fig = go.Figure()
        
        # Color based on transaction type and create labels
        colors = []
        labels = []
        for _, row in largest_transactions.iterrows():
            colors.append('green' if row['type'] == 'income' else 'red')
            note = str(row['note']) if row['note'] else 'No description'
            truncated_note = (note[:40] + '...') if len(note) > 40 else note
            labels.append(f"{truncated_note} (${abs(row['amount']):.2f})")
        
        fig.add_trace(go.Bar(
            y=labels,
            x=largest_transactions['amount'].abs(),
            orientation='h',
            marker_color=colors,
            text=[f"${abs(amount):.2f}" for amount in largest_transactions['amount']],
            textposition='auto',
            hovertemplate=(
                "<b>%{y}</b><br>" +
                "Amount: $%{x:.2f}<br>" +
                "Category: " + largest_transactions['category'] + "<br>" +
                "Type: " + largest_transactions['type'] + "<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=f"Largest Transactions: {query.title()}",
            xaxis_title="Amount ($)",
            yaxis_title="Transactions",
            height=600,
            showlegend=False,
            yaxis=dict(autorange="reversed")  # Largest at top
        )
        
        # Calculate insights
        total_largest = largest_transactions['amount'].sum()
        avg_large = largest_transactions['amount'].mean()
        largest_single = largest_transactions['amount'].max()
        smallest_large = largest_transactions['amount'].min()
        
        income_count = len(largest_transactions[largest_transactions['type'] == 'income'])
        expense_count = len(largest_transactions[largest_transactions['type'] == 'expense'])
        
        insights = [
            f"💰 Largest transaction: ${largest_single:.2f}",
            f"📊 Smallest in top 10: ${smallest_large:.2f}",
            f"📈 Average of top {len(largest_transactions)}: ${avg_large:.2f}",
            f"🎯 Total shown: ${total_largest:.2f}",
            f"💵 {income_count} income, {expense_count} expense transactions"
        ]
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'bar',
            'title': f"Largest Transactions: {query.title()}",
            'analysis_notes': f"Shows the {len(largest_transactions)} largest individual transactions from {time_period}",
            'insights': insights,
            'data_points': len(largest_transactions),
            'filters_applied': {'time_period': time_period, 'limit': 10}
        }
    
    def _create_income_expense_trend_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create a line chart showing income vs expenses over time"""
        df = pd.DataFrame(transactions)
        
        if df.empty or 'date' not in df.columns:
            return self._create_empty_chart("No date data available for trend analysis")
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by month and type
        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly_data = df.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        
        # Ensure we have both income and expense columns
        if 'income' not in monthly_data.columns:
            monthly_data['income'] = 0
        if 'expense' not in monthly_data.columns:
            monthly_data['expense'] = 0
        
        # Create line chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            name='Income',
            x=monthly_data.index,
            y=monthly_data['income'],
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate="<b>Income</b><br>Month: %{x}<br>Amount: $%{y:.2f}<extra></extra>"
        ))
        
        fig.add_trace(go.Scatter(
            name='Expenses',
            x=monthly_data.index,
            y=monthly_data['expense'],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=8, symbol='square'),
            hovertemplate="<b>Expenses</b><br>Month: %{x}<br>Amount: $%{y:.2f}<extra></extra>"
        ))
        
        # Add net savings line (income - expenses)
        monthly_data['net'] = monthly_data['income'] - monthly_data['expense']
        fig.add_trace(go.Scatter(
            name='Net Savings',
            x=monthly_data.index,
            y=monthly_data['net'],
            mode='lines+markers',
            line=dict(color='blue', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate="<b>Net Savings</b><br>Month: %{x}<br>Amount: $%{y:.2f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Income vs Expenses Over Time",
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        # Calculate insights
        total_income = monthly_data['income'].sum()
        total_expenses = monthly_data['expense'].sum()
        net_savings = total_income - total_expenses
        savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
        
        # Find best and worst months
        best_month = monthly_data['net'].idxmax()
        best_amount = monthly_data['net'].max()
        worst_month = monthly_data['net'].idxmin()
        worst_amount = monthly_data['net'].min()
        
        insights = [
            f"💰 Total Income: ${total_income:.2f}",
            f"📊 Total Expenses: ${total_expenses:.2f}",
            f"🎯 Net Savings: ${net_savings:.2f}",
            f"📈 Savings Rate: {savings_rate:.1f}%",
            f"🕐 Period: {len(monthly_data)} months",
            f"⭐ Best month: {best_month} (${best_amount:.2f})",
            f"⚠️ Worst month: {worst_month} (${worst_amount:.2f})"
        ]
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'line',
            'title': f"Income vs Expenses Over Time",
            'analysis_notes': "Shows monthly income, expenses, and net savings trends. Positive net savings (blue line) means you saved money that month.",
            'insights': insights,
            'data_points': len(df),
            'filters_applied': {'time_period': 'all_time'}
        }
    
    def _create_category_comparison_chart(self, transactions: List[Dict], categories: List[str], query: str) -> Dict[str, Any]:
        """Create a bar chart comparing specific categories"""
        df = pd.DataFrame(transactions)
        
        if df.empty:
            return self._create_empty_chart("No transaction data available")
        
        # Filter for expense transactions in the specified categories
        expense_df = df[df['type'] == 'expense']
        if expense_df.empty:
            return self._create_empty_chart("No expense data available for comparison")
        
        # Find matching categories (case insensitive)
        category_totals = {}
        available_categories = expense_df['category'].str.lower().unique()
        
        for category in categories:
            # Find matching category (case insensitive)
            matching_cats = [cat for cat in available_categories if category.lower() in cat.lower()]
            if matching_cats:
                # Use the first matching category
                actual_category = matching_cats[0]
                total = expense_df[expense_df['category'].str.lower() == actual_category]['amount'].sum()
                category_totals[category] = total
            else:
                category_totals[category] = 0
        
        # Create the bar chart
        fig = go.Figure()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Red, Teal, Blue, Green
        
        for i, (category, amount) in enumerate(category_totals.items()):
            fig.add_trace(go.Bar(
                name=category,
                x=[category],
                y=[amount],
                marker_color=colors[i % len(colors)],
                text=[f"${amount:.2f}"],
                textposition='auto',
                textfont=dict(size=14, color='white', weight='bold')
            ))
        
        # Calculate insights
        insights = []
        if len(category_totals) >= 2:
            cat_names = list(category_totals.keys())
            amounts = list(category_totals.values())
            
            insights.append(f"📊 {cat_names[0]}: ${amounts[0]:.2f}")
            insights.append(f"📊 {cat_names[1]}: ${amounts[1]:.2f}")
            
            if amounts[0] > amounts[1]:
                difference = amounts[0] - amounts[1]
                percentage = (difference / amounts[1]) * 100 if amounts[1] > 0 else 100
                insights.append(f"🎯 {cat_names[0]} is ${difference:.2f} higher ({percentage:.1f}% more)")
                higher_cat, lower_cat = cat_names[0], cat_names[1]
            else:
                difference = amounts[1] - amounts[0]
                percentage = (difference / amounts[0]) * 100 if amounts[0] > 0 else 100
                insights.append(f"🎯 {cat_names[1]} is ${difference:.2f} higher ({percentage:.1f}% more)")
                higher_cat, lower_cat = cat_names[1], cat_names[0]
            
            insights.append(f"💰 Total compared: ${sum(amounts):.2f}")
            
            analysis_notes = f"Direct comparison between {cat_names[0]} and {cat_names[1]} spending. {higher_cat} has higher total spending."
        else:
            analysis_notes = f"Comparison of {list(category_totals.keys())[0]} spending"
            insights = [f"Found ${list(category_totals.values())[0]:.2f} in {list(category_totals.keys())[0]} spending"]
        
        fig.update_layout(
            title=f"Spending Comparison: {query.title()}",
            xaxis_title="Category",
            yaxis_title="Amount ($)",
            showlegend=False,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'bar',
            'title': f"Spending Comparison: {query.title()}",
            'analysis_notes': analysis_notes,
            'insights': insights,
            'data_points': len(expense_df[expense_df['category'].str.lower().isin([cat.lower() for cat in categories])]),
            'filters_applied': {'categories': categories}
        }
    
    def _fallback_chart_generation(self, query: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Fallback chart generation when Gemini is unavailable"""
        # First try special queries
        special_result = self._handle_special_queries(query, transactions)
        if special_result:
            return special_result
        
        # Then try other chart types
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['category', 'categories', 'pie', 'distribution']):
            return self._create_category_pie_chart(transactions, query)
        elif any(word in query_lower for word in ['time', 'trend', 'line', 'over time']):
            return self._create_trend_line_chart(transactions, query)
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'bar']):
            return self._create_comparison_bar_chart(transactions, query)
        elif any(word in query_lower for word in ['income', 'expense', 'balance']):
            return self._create_income_expense_chart(transactions, query)
        else:
            return self._create_category_pie_chart(transactions, query)
    
    def _prepare_chart_data_summary(self, transactions: List[Dict]) -> str:
        """Prepare transaction summary for LLM analysis"""
        if not transactions:
            return "No transaction data available."
        
        df = pd.DataFrame(transactions)
        
        summary = f"""
        Transaction Overview:
        - Total transactions: {len(transactions)}
        - Time period: {df['date'].min() if 'date' in df.columns else 'Unknown'} to {df['date'].max() if 'date' in df.columns else 'Unknown'}
        - Transaction types: {df['type'].value_counts().to_dict() if 'type' in df.columns else 'N/A'}
        
        Category Distribution:
        {df['category'].value_counts().to_dict() if 'category' in df.columns else 'No categories'}
        
        Amount Statistics:
        - Total amount: ${df['amount'].sum():.2f}
        - Average amount: ${df['amount'].mean():.2f}
        - Min amount: ${df['amount'].min():.2f}
        - Max amount: ${df['amount'].max():.2f}
        """
        
        return summary
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract chart configuration"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                config = json.loads(json_match.group())
                # Ensure required fields
                if 'chart_type' not in config:
                    config['chart_type'] = 'bar'
                if 'title' not in config:
                    config['title'] = 'Financial Chart'
                if 'analysis_notes' not in config:
                    config['analysis_notes'] = 'Chart generated from your data'
                return config
            else:
                # Fallback configuration
                return {
                    "chart_type": "bar",
                    "title": "Transaction Overview",
                    "x_axis": "category",
                    "y_axis": "amount",
                    "color_by": "type",
                    "filters": {"time_period": "all_time"},
                    "analysis_notes": "Default chart showing transaction amounts by category",
                    "insights": ["Add more specific queries for better charts"]
                }
        except Exception as e:
            logging.error(f"Failed to parse LLM response: {e}")
            return {
                "chart_type": "bar",
                "title": "Transaction Overview",
                "x_axis": "category", 
                "y_axis": "amount",
                "analysis_notes": "Fallback chart",
                "insights": ["AI analysis temporarily unavailable"]
            }
    
    def _create_chart_from_config(self, config: Dict, transactions: List[Dict]) -> Dict[str, Any]:
        """Create actual chart based on configuration"""
        try:
            df = pd.DataFrame(transactions)
            
            if df.empty:
                return self._create_empty_chart("No data available for chart generation")
            
            chart_type = config.get('chart_type', 'bar')
            title = config.get('title', 'Financial Chart')
            
            if chart_type == 'pie':
                fig = self._create_pie_chart(df, config)
            elif chart_type == 'line':
                fig = self._create_line_chart(df, config)
            elif chart_type == 'scatter':
                fig = self._create_scatter_chart(df, config)
            else:  # bar chart default
                fig = self._create_bar_chart(df, config)
            
            fig.update_layout(
                title=title, 
                height=500,
                showlegend=True
            )
            
            return {
                'chart_json': fig.to_json(),
                'chart_type': chart_type,
                'title': title,
                'analysis_notes': config.get('analysis_notes', ''),
                'insights': config.get('insights', ['Chart generated successfully']),
                'data_points': len(df),
                'filters_applied': config.get('filters', {})
            }
            
        except Exception as e:
            logging.error(f"Chart creation failed: {e}")
            return {'error': f'Chart creation failed: {str(e)}'}
    
    def _create_pie_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create pie chart"""
        expenses = df[df['type'] == 'expense']
        if expenses.empty:
            expenses = df
            
        category_totals = expenses.groupby('category')['amount'].sum()
        
        fig = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title=config.get('title', 'Expenses by Category')
        )
        return fig
    
    def _create_bar_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create bar chart"""
        # Aggregate by category and type
        category_totals = df.groupby(['category', 'type'])['amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure()
        if 'income' in category_totals.columns:
            fig.add_trace(go.Bar(
                name='Income', 
                x=category_totals.index, 
                y=category_totals['income'], 
                marker_color='green'
            ))
        if 'expense' in category_totals.columns:
            fig.add_trace(go.Bar(
                name='Expense', 
                x=category_totals.index, 
                y=category_totals['expense'], 
                marker_color='red'
            ))
        
        fig.update_layout(
            title=config.get('title', 'Transactions by Category'), 
            barmode='group',
            xaxis_title="Category",
            yaxis_title="Amount ($)"
        )
        return fig
    
    def _create_line_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create line chart for trends"""
        if 'date' not in df.columns:
            # Fallback to bar chart if no dates
            return self._create_bar_chart(df, config)
        
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        
        # Aggregate by date
        daily_totals = df_sorted.groupby('date')['amount'].sum().reset_index()
        
        fig = px.line(
            daily_totals, 
            x='date', 
            y='amount', 
            title=config.get('title', 'Spending Over Time')
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount ($)"
        )
        return fig
    
    def _create_scatter_chart(self, df: pd.DataFrame, config: Dict) -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(
            df, 
            x='date' if 'date' in df.columns else 'category', 
            y='amount', 
            color='type', 
            title=config.get('title', 'Transaction Distribution')
        )
        return fig
    
    def _create_category_pie_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create category pie chart for fallback"""
        df = pd.DataFrame(transactions)
        
        if df.empty:
            return self._create_empty_chart("No data available")
        
        expenses = df[df['type'] == 'expense']
        if expenses.empty:
            return self._create_empty_chart("No expense data available")
        
        category_totals = expenses.groupby('category')['amount'].sum()
        
        fig = px.pie(
            values=category_totals.values,
            names=category_totals.index,
            title=f"Spending by Category: {query}"
        )
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'pie',
            'title': f"Spending by Category: {query}",
            'analysis_notes': f"Shows your spending distribution across {len(category_totals)} categories",
            'insights': [
                f"Top category: {category_totals.idxmax()}",
                f"Total expenses: ${expenses['amount'].sum():.2f}",
                "Consider setting budgets for top spending categories"
            ],
            'data_points': len(expenses),
            'filters_applied': {}
        }
    
    def _create_trend_line_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create trend line chart for fallback"""
        df = pd.DataFrame(transactions)
        
        if df.empty or 'date' not in df.columns:
            return self._create_empty_chart("No date data available for trends")
        
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values('date')
        daily_totals = df_sorted.groupby('date')['amount'].sum().reset_index()
        
        fig = px.line(
            daily_totals, 
            x='date', 
            y='amount',
            title=f"Spending Trends: {query}"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Amount ($)"
        )
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'line',
            'title': f"Spending Trends: {query}",
            'analysis_notes': f"Shows spending patterns over {len(daily_totals)} days",
            'insights': [
                f"Time period: {daily_totals['date'].min().strftime('%Y-%m-%d')} to {daily_totals['date'].max().strftime('%Y-%m-%d')}",
                f"Total transactions: {len(df)}",
                "Look for weekly or monthly patterns in your spending"
            ],
            'data_points': len(daily_totals),
            'filters_applied': {}
        }
    
    def _create_comparison_bar_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create comparison bar chart for fallback"""
        return self._create_category_comparison_chart(transactions, ['Food', 'Entertainment'], query)
    
    def _create_income_expense_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create income vs expense chart for fallback"""
        df = pd.DataFrame(transactions)
        
        if df.empty:
            return self._create_empty_chart("No data available")
        
        # Monthly income vs expenses
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.to_period('M').astype(str)
            monthly_data = df.groupby(['month', 'type'])['amount'].sum().unstack(fill_value=0)
        else:
            # If no dates, use overall totals
            type_totals = df.groupby('type')['amount'].sum()
            monthly_data = pd.DataFrame({
                'Overall': type_totals
            }).T
        
        fig = go.Figure()
        if 'income' in monthly_data.columns:
            fig.add_trace(go.Bar(
                name='Income', 
                x=monthly_data.index, 
                y=monthly_data['income'],
                marker_color='green'
            ))
        if 'expense' in monthly_data.columns:
            fig.add_trace(go.Bar(
                name='Expense', 
                x=monthly_data.index, 
                y=monthly_data['expense'],
                marker_color='red'
            ))
        
        fig.update_layout(
            title=f"Income vs Expenses: {query}",
            barmode='group',
            xaxis_title="Period",
            yaxis_title="Amount ($)"
        )
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'bar',
            'title': f"Income vs Expenses: {query}",
            'analysis_notes': "Shows the balance between your income and expenses",
            'insights': [
                f"Total income: ${monthly_data['income'].sum() if 'income' in monthly_data.columns else 0:.2f}",
                f"Total expenses: ${monthly_data['expense'].sum() if 'expense' in monthly_data.columns else 0:.2f}",
                f"Net savings: ${(monthly_data['income'].sum() - monthly_data['expense'].sum()) if 'income' in monthly_data.columns and 'expense' in monthly_data.columns else 0:.2f}"
            ],
            'data_points': len(df),
            'filters_applied': {}
        }
    
    def _create_overview_chart(self, transactions: List[Dict], query: str) -> Dict[str, Any]:
        """Create overview chart for general queries"""
        return self._create_category_pie_chart(transactions, query)
    
    def _create_empty_chart(self, message: str) -> Dict[str, Any]:
        """Create an empty chart with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=message,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        
        return {
            'chart_json': fig.to_json(),
            'chart_type': 'message',
            'title': message,
            'analysis_notes': 'No data available for chart generation',
            'insights': ['Add transactions to generate charts'],
            'data_points': 0,
            'filters_applied': {}
        }