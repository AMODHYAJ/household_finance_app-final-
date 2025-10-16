import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class ChartCreatorAgent:
    """
    Fixed chart creator with proper JSON serialization
    """

    def __init__(self, data_agent):
        self.data_agent = data_agent

    def _prepare_transaction_data(self, transactions):
        """Prepare and normalize transaction data with proper field names"""
        if not transactions:
            print("❌ No transactions provided to chart creator")
            return None

        try:
            # Create DataFrame with consistent field names
            transaction_list = []
            for t in transactions:
                # Handle both 'type' and 't_type' field names
                transaction_type = getattr(t, "type", None) or getattr(
                    t, "t_type", "expense"
                )

                transaction_list.append(
                    {
                        "date": t.date,
                        "type": str(transaction_type).lower().strip(),
                        "category": (
                            str(t.category).strip().title()
                            if t.category
                            else "Uncategorized"
                        ),
                        "amount": float(t.amount) if t.amount else 0.0,
                        "note": t.note or "",
                    }
                )

            df = pd.DataFrame(transaction_list)

            print(f"📊 Prepared {len(df)} transactions for charting")

            if not df.empty:
                # Debug: Show expense data specifically
                expense_df = df[df["type"] == "expense"]
                if not expense_df.empty:
                    print("💰 Expense data summary:")
                    expense_summary = (
                        expense_df.groupby("category")["amount"]
                        .agg(["sum", "count"])
                        .round(2)
                    )
                    print(expense_summary)
                    print(f"💵 Total expenses: ${expense_df['amount'].sum():.2f}")

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            return df

        except Exception as e:
            print(f"❌ Error preparing transaction data: {e}")
            import traceback

            traceback.print_exc()
            return None

    def create_expenses_by_category_chart(self, transactions):
        """Create expenses by category pie chart with proper data filtering"""
        print("🎯 Creating expenses by category pie chart...")
        df = self._prepare_transaction_data(transactions)
        if df is None or df.empty:
            return self._create_empty_chart("No transaction data available")

        try:
            # Get only expense transactions
            expenses_df = df[df["type"].str.lower() == "expense"]
            print(f"📊 Found {len(expenses_df)} expense transactions after filtering")

            if expenses_df.empty:
                print("❌ No expense transactions after filtering")
                return self._create_empty_chart("No expense transactions found")

            # Calculate category totals - SUM of amounts
            category_totals = (
                expenses_df.groupby("category")["amount"].sum().reset_index()
            )
            category_totals = category_totals[category_totals["amount"] > 0]
            category_totals = category_totals.sort_values("amount", ascending=False)

            total_expenses = category_totals["amount"].sum()
            print(f"📊 Total expenses calculated: ${total_expenses:.2f}")
            print(f"📊 Category amounts:")
            for _, row in category_totals.iterrows():
                print(f"   - {row['category']}: ${row['amount']:.2f}")

            # Convert to regular Python lists to avoid numpy serialization issues
            categories = category_totals["category"].tolist()
            amounts = category_totals["amount"].tolist()

            # Create pie chart with explicit data
            fig = go.Figure()

            fig.add_trace(
                go.Pie(
                    labels=categories,
                    values=amounts,
                    hole=0.3,
                    marker=dict(colors=px.colors.qualitative.Set3),
                    hovertemplate="<b>%{label}</b><br>Amount: $%{value:.2f}<br>Percentage: %{percent}<extra></extra>",
                    textinfo="label+percent",
                    textposition="inside",
                )
            )

            fig.update_layout(
                title=f"Expenses by Category<br><sub>Total Expenses: ${total_expenses:.2f}</sub>",
                showlegend=True,
                height=500,
                annotations=[
                    dict(
                        text=f"Total<br>${total_expenses:.2f}",
                        x=0.5,
                        y=0.5,
                        font_size=14,
                        showarrow=False,
                    )
                ],
            )

            print("✅ Expenses pie chart created successfully")
            return fig

        except Exception as e:
            print(f"❌ Expenses pie chart error: {e}")
            import traceback

            traceback.print_exc()
            return self._create_error_chart(f"Error creating expenses chart: {str(e)}")

    def create_expenses_bar_chart(self, transactions):
        """Create horizontal bar chart for expenses by category"""
        print("🎯 Creating expenses by category bar chart...")
        df = self._prepare_transaction_data(transactions)
        if df is None or df.empty:
            return self._create_empty_chart("No transaction data available")

        try:
            # Get only expense transactions
            expenses_df = df[df["type"].str.lower() == "expense"]
            print(f"📊 Found {len(expenses_df)} expense transactions for bar chart")

            if expenses_df.empty:
                return self._create_empty_chart("No expense transactions found")

            # Calculate category totals
            category_totals = (
                expenses_df.groupby("category")["amount"].sum().reset_index()
            )
            category_totals = category_totals[category_totals["amount"] > 0]
            category_totals = category_totals.sort_values("amount", ascending=True)

            total_expenses = category_totals["amount"].sum()

            # Convert to regular Python lists
            categories = category_totals["category"].tolist()
            amounts = category_totals["amount"].tolist()
            percentages = [(amt / total_expenses * 100) for amt in amounts]

            # Create horizontal bar chart
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    y=categories,
                    x=amounts,
                    orientation="h",
                    marker_color="#e74c3c",
                    hovertemplate="<b>%{y}</b><br>Amount: $%{x:.2f}<br>Percentage: %{customdata:.1f}%<extra></extra>",
                    customdata=percentages,
                    text=[f"${amt:.2f}" for amt in amounts],
                    textposition="auto",
                )
            )

            fig.update_layout(
                title=f"Expenses by Category - Bar Chart<br><sub>Total: ${total_expenses:.2f}</sub>",
                xaxis_title="Amount ($)",
                yaxis_title="Categories",
                showlegend=False,
                height=max(400, len(categories) * 40),
                bargap=0.2,
            )

            print("✅ Expenses bar chart created successfully")
            return fig

        except Exception as e:
            print(f"❌ Expenses bar chart error: {e}")
            return self._create_error_chart(
                f"Error creating expenses bar chart: {str(e)}"
            )

    def create_category_comparison_chart(self, transactions):
        """Create a comparison chart showing both transaction count and amount"""
        print("🎯 Creating category comparison chart...")
        df = self._prepare_transaction_data(transactions)
        if df is None or df.empty:
            return self._create_empty_chart("No transaction data available")

        try:
            # Get expense transactions
            expenses_df = df[df["type"].str.lower() == "expense"]

            if expenses_df.empty:
                return self._create_empty_chart("No expense transactions found")

            # Calculate both count and sum for each category
            category_stats = (
                expenses_df.groupby("category")
                .agg({"amount": ["sum", "count"]})
                .round(2)
            )

            # Flatten column names
            category_stats.columns = ["total_amount", "transaction_count"]
            category_stats = category_stats.reset_index()
            category_stats = category_stats.sort_values("total_amount", ascending=False)

            print("📊 Category Comparison Stats:")
            print(category_stats)

            # Create subplot with two bars
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    "Total Amount by Category",
                    "Transaction Count by Category",
                ),
                specs=[[{"type": "bar"}, {"type": "bar"}]],
            )

            # Amount bar chart
            fig.add_trace(
                go.Bar(
                    x=category_stats["category"],
                    y=category_stats["total_amount"],
                    name="Total Amount",
                    marker_color="#e74c3c",
                    hovertemplate="<b>%{x}</b><br>Amount: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Count bar chart
            fig.add_trace(
                go.Bar(
                    x=category_stats["category"],
                    y=category_stats["transaction_count"],
                    name="Transaction Count",
                    marker_color="#3498db",
                    hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            fig.update_layout(
                title="Category Comparison: Amount vs Transaction Count",
                height=500,
                showlegend=False,
            )

            fig.update_xaxes(tickangle=45, row=1, col=1)
            fig.update_xaxes(tickangle=45, row=1, col=2)
            fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=2)

            print("✅ Category comparison chart created successfully")
            return fig

        except Exception as e:
            print(f"❌ Category comparison chart error: {e}")
            return self._create_error_chart(
                f"Error creating comparison chart: {str(e)}"
            )

    def create_income_vs_expenses_chart(self, transactions):
        """Create income vs expenses chart with proper scaling"""
        print("🎯 Creating income vs expenses chart...")
        df = self._prepare_transaction_data(transactions)
        if df is None or df.empty:
            return self._create_empty_chart("No transaction data available")

        try:
            # Create monthly aggregates
            df["month"] = df["date"].dt.to_period("M")
            monthly_data = (
                df.groupby(["month", "type"])["amount"].sum().unstack(fill_value=0)
            )

            print(f"📊 Monthly data columns: {monthly_data.columns.tolist()}")
            print(f"📊 Monthly data:\n{monthly_data}")

            # Ensure required columns exist
            for col in ["income", "expense"]:
                if col not in monthly_data.columns:
                    monthly_data[col] = 0

            monthly_data.index = monthly_data.index.astype(str)

            total_income = monthly_data["income"].sum()
            total_expenses = monthly_data["expense"].sum()
            print(f"📊 Total income: ${total_income:.2f}")
            print(f"📊 Total expenses: ${total_expenses:.2f}")

            # Convert to regular lists to avoid scaling issues
            months = monthly_data.index.tolist()
            income_values = monthly_data["income"].tolist()
            expense_values = monthly_data["expense"].tolist()
            net_values = [inc - exp for inc, exp in zip(income_values, expense_values)]

            fig = go.Figure()

            # Add income bars
            if total_income > 0:
                fig.add_trace(
                    go.Bar(
                        name="Income",
                        x=months,
                        y=income_values,
                        marker_color="#2ecc71",
                        hovertemplate="<b>%{x}</b><br>Income: $%{y:,.2f}<extra></extra>",
                    )
                )

            # Add expense bars
            if total_expenses > 0:
                fig.add_trace(
                    go.Bar(
                        name="Expenses",
                        x=months,
                        y=expense_values,
                        marker_color="#e74c3c",
                        hovertemplate="<b>%{x}</b><br>Expenses: $%{y:,.2f}<extra></extra>",
                    )
                )

            # Add net savings line
            fig.add_trace(
                go.Scatter(
                    name="Net Savings",
                    x=months,
                    y=net_values,
                    mode="lines+markers",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=8),
                    hovertemplate="<b>%{x}</b><br>Net: $%{y:,.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                title="Income vs Expenses Over Time",
                barmode="group",
                xaxis_title="Month",
                yaxis_title="Amount ($)",
                hovermode="x unified",
                height=500,
                showlegend=True,
            )

            print("✅ Income vs expenses chart created successfully")
            return fig

        except Exception as e:
            print(f"❌ Income vs Expenses chart error: {e}")
            import traceback

            traceback.print_exc()
            return self._create_error_chart(
                f"Error creating income vs expenses chart: {str(e)}"
            )

    def _create_empty_chart(self, message):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            title="No Data Available",
        )
        return fig

    def _create_error_chart(self, message):
        """Create error chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {message}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            title="Chart Error",
        )
        return fig

    def create_spending_timeline_chart(self, transactions):
        """Create spending timeline chart with proper daily aggregation"""
        print("🎯 Creating spending timeline chart...")

        # Prepare data manually to ensure proper aggregation
        expense_data = []
        for t in transactions:
            transaction_type = getattr(t, "type", None) or getattr(t, "t_type", "expense")
            if str(transaction_type).lower() == "expense":
                expense_data.append(
                    {
                        "date": t.date,
                        "amount": float(t.amount) if t.amount else 0.0,
                        "category": t.category,
                        "note": t.note,
                   }
                )

        if not expense_data:
            return self._create_empty_chart("No expense transactions found")

        try:
            # Create DataFrame and ensure proper date handling
            df = pd.DataFrame(expense_data)
            df["date"] = pd.to_datetime(df["date"])

            # Group by date and sum amounts - this should give us daily totals
            daily_expenses = df.groupby("date")["amount"].sum().reset_index()
            daily_expenses = daily_expenses.sort_values("date")

            print(f"📊 Daily expenses timeline - {len(daily_expenses)} days")
            print("📊 Daily amounts:")
            for _, row in daily_expenses.iterrows():
                print(f"   - {row['date'].strftime('%Y-%m-%d')}: ${row['amount']:.2f}")

            print(f"📊 Timeline total: ${daily_expenses['amount'].sum():.2f}")
            print(f"📊 Timeline max daily: ${daily_expenses['amount'].max():.2f}")
            print(f"📊 Timeline min daily: ${daily_expenses['amount'].min():.2f}")

            # Convert to regular Python lists
            dates = daily_expenses["date"].dt.strftime("%Y-%m-%d").tolist()
            amounts = daily_expenses["amount"].tolist()

            # Create timeline chart
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=amounts,
                    mode="lines+markers",
                    line=dict(color="#e74c3c", width=3),
                    marker=dict(size=8, color="#e74c3c"),
                    name="Daily Spending",
                    hovertemplate="<b>%{x}</b><br>Amount: $%{y:.2f}<extra></extra>",
                )
            )

            # Add 7-day moving average if we have enough data
            if len(daily_expenses) > 7:
                moving_avg = daily_expenses["amount"].rolling(window=7).mean().tolist()
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=moving_avg,
                        mode="lines",
                        line=dict(color="#3498db", width=3, dash="dash"),
                        name="7-Day Average",
                        hovertemplate="<b>%{x}</b><br>7-Day Avg: $%{y:.2f}<extra></extra>",
                    )
                )

            # Calculate some stats for the title
            total_spent = sum(amounts)
            avg_daily = total_spent / len(amounts) if amounts else 0
            max_daily = max(amounts) if amounts else 0

            fig.update_layout(
                title=f"Spending Timeline<br><sub>Total: ${total_spent:.2f} | Avg Daily: ${avg_daily:.2f} | Max: ${max_daily:.2f}</sub>",
                xaxis_title="Date",
                yaxis_title="Amount ($)",
                hovermode="x unified",
                height=500,
                showlegend=len(daily_expenses)
                > 7,  # Only show legend if we have moving average
            )

            print("✅ Spending timeline chart created successfully")
            return fig

        except Exception as e:
            print(f"❌ Spending timeline chart error: {e}")
            import traceback

            traceback.print_exc()
            return self._create_error_chart(f"Error creating timeline chart: {str(e)}")
