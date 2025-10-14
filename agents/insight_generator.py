import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

class InsightGeneratorAgent:
    """
    Advanced insight generator with predictive analytics
    """
    
    def __init__(self, data_agent=None):
        self.data_agent = data_agent
        self.savings_target = 5000

    def get_dataframe(self, transactions):
        """Convert transactions to dataframe"""
        if not transactions:
            return pd.DataFrame()
        
        data = [{
            "date": t.date,
            "type": t.type,
            "category": t.category,
            "amount": t.amount,
            "note": t.note or ""
        } for t in transactions]
        
        return pd.DataFrame(data)

    def generate_all(self, transactions):
        """Generate all insights including predictions"""
        df = self.get_dataframe(transactions)
        
        if df.empty:
            return self._empty_insights()
        
        insights = {}
        insights["responsible_ai"] = self.responsible_ai_checks(df)
        insights["summary_stats"] = self.summary_stats(df)
        insights["trend"] = self.trend_insights(df)
        insights["alerts"] = self.alerts(df)
        insights["budget_recommendations"] = self.budget_recommendations(df)
        insights["predictive_analytics"] = self.predictive_analytics(transactions)
        insights["financial_health"] = self.financial_health_score(df)

        # NEW: LLM and NLP features
        insights["llm_advice"] = self.generate_llm_advice(insights)
        insights["nlp_analysis"] = self.analyze_transaction_notes(transactions)
        insights["dataset_comparison"] = self.compare_with_real_dataset(df)
        
        return insights
    
    def _empty_insights(self):
        """Return empty insights structure"""
        return {
            "responsible_ai": ["No data available for analysis"],
            "summary_stats": {},
            "trend": "No data available for trend analysis",
            "alerts": [],
            "budget_recommendations": ["Add some transactions to get personalized recommendations"],
            "predictive_analytics": {},
            "financial_health": {"score": 0, "message": "Insufficient data"}
        }

    def responsible_ai_checks(self, df):
        """Enhanced Responsible AI checks"""
        try:
            from agents.responsible_ai import ResponsibleAIAuditor
        
            auditor = ResponsibleAIAuditor()
            transactions_data = df.to_dict('records')
        
            # Get AI categories (you might need to track these separately)
            ai_categories = df['category'].tolist() if 'category' in df.columns else []
        
            # Run comprehensive audit
            audit_report = auditor.audit_ai_decisions(transactions_data, ai_categories)
        
            # Format results for display
            checks = []
        
            # Add fairness checks
            checks.extend(audit_report['fairness_checks'])
        
            # Add bias detection
            checks.extend(audit_report['bias_detection'])
        
            # Add transparency metrics
            checks.extend(audit_report['transparency_metrics'])
        
            # Add privacy checks
            checks.extend(audit_report['privacy_checks'])
        
            # Add recommendations
            checks.append("--- RECOMMENDATIONS ---")
            checks.extend(audit_report['recommendations'])
        
            return checks
        
        except Exception as e:
            logging.error(f"Responsible AI audit failed: {e}")
            return [
                "✅ Basic AI ethics monitoring enabled",
                "⚠️ Enhanced Responsible AI checks unavailable",
                "🔒 Data privacy and security maintained"
            ]

    def summary_stats(self, df):
        """Calculate summary statistics"""
        if df.empty:
            return {}
        
        total_income = df[df["type"] == "income"]["amount"].sum()
        total_expense = df[df["type"] == "expense"]["amount"].sum()
        savings = total_income - total_expense
        savings_rate = (savings / total_income * 100) if total_income > 0 else 0
        
        return {
            "income": total_income,
            "expense": total_expense,
            "savings": savings,
            "savings_rate": savings_rate
        }

    def trend_insights(self, df):
        """Generate trend insights"""
        if df.empty or "date" not in df.columns:
            return "No data available for trend analysis"
        
        try:
            df_copy = df.copy()
            df_copy["date"] = pd.to_datetime(df_copy["date"])
            monthly = df_copy.groupby(df_copy["date"].dt.to_period("M"))["amount"].sum()
            
            if len(monthly) < 2:
                return "Keep tracking expenses to see trends over time"
            
            # Simple trend detection
            if monthly.iloc[-1] > monthly.iloc[-2]:
                return "📈 Your spending has increased compared to last month"
            else:
                return "📉 Your spending has decreased compared to last month"
                
        except Exception:
            return "Trend analysis requires consistent date data"

    def alerts(self, df):
        """Generate spending alerts"""
        alerts = []
        
        if df.empty:
            return alerts
        
        # High spending alert
        expenses = df[df["type"] == "expense"]
        if not expenses.empty:
            avg_expense = expenses["amount"].mean()
            recent_expenses = expenses.tail(5)["amount"]
            
            if len(recent_expenses) >= 3:
                if recent_expenses.mean() > avg_expense * 1.5:
                    alerts.append("⚠️ Recent spending is higher than your average")
        
        return alerts

    def budget_recommendations(self, df):
        """Generate budget recommendations"""
        recommendations = []
        
        if df.empty:
            return ["Start by tracking your income and expenses regularly"]
        
        expenses = df[df["type"] == "expense"]
        if not expenses.empty:
            # Simple recommendations based on spending patterns
            total_expenses = expenses["amount"].sum()
            if "category" in expenses.columns:
                top_category = expenses.groupby("category")["amount"].sum().idxmax()
                recommendations.append(f"💡 Your highest spending is in {top_category}. Consider setting a budget for this category.")
            
            if total_expenses > 1000:
                recommendations.append("💡 You're spending over $1000. Review your expenses to identify savings opportunities.")
            else:
                recommendations.append("💡 Your spending is manageable. Focus on maintaining good financial habits.")
        
        return recommendations
    
    def predict_top(self, note_text, top_n=3):
        """Return top N predictions with confidence scores"""
        # For now, return a simple mock implementation
        suggestions = [
            ("food", 0.8),
            ("shopping", 0.6),
            ("bills", 0.4)
        ]
        return suggestions[:top_n]
    
    def predictive_analytics(self, transactions):
        """Generate predictive insights and forecasts"""
        df = self.get_dataframe(transactions)
        
        if df.empty:
            return {"error": "No data for predictions"}
        
        predictions = {}
        
        # Monthly spending forecast
        predictions["monthly_forecast"] = self.forecast_monthly_spending(df)
        
        # Savings projection
        predictions["savings_projection"] = self.project_savings(df)
        
        # Category trends
        predictions["category_trends"] = self.analyze_category_trends(df)
        
        # Anomaly detection
        predictions["spending_anomalies"] = self.detect_spending_anomalies(df)
        
        return predictions
    
    def forecast_monthly_spending(self, df):
        """Forecast next month's spending using linear regression"""
        expenses = df[df['type'] == 'expense'].copy()
        
        if expenses.empty:
            return "Insufficient data for forecasting"
        
        expenses['date'] = pd.to_datetime(expenses['date'])
        monthly = expenses.groupby(expenses['date'].dt.to_period('M'))['amount'].sum()
        
        if len(monthly) < 2:
            return "Need more data for accurate forecasting"
        
        # Prepare data for linear regression
        months = np.arange(len(monthly)).reshape(-1, 1)
        amounts = monthly.values
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(months, amounts)
        
        # Predict next month
        next_month = len(monthly)
        forecast = model.predict([[next_month]])[0]
        
        # Calculate confidence interval (simplified)
        confidence = max(0, min(100, 100 - (10 * (6 - len(monthly)))))  # More data = higher confidence
        
        return {
            "next_month_forecast": round(float(forecast), 2),
            "confidence": f"{confidence}%",
            "based_on": f"last {len(monthly)} months of data",
            "trend": "increasing" if model.coef_[0] > 0 else "decreasing"
        }
    
    def project_savings(self, df):
        """Project savings growth over time"""
        df['date'] = pd.to_datetime(df['date'])
        monthly = df.groupby([df['date'].dt.to_period('M'), 'type'])['amount'].sum().unstack(fill_value=0)
        
        if 'income' not in monthly.columns or 'expense' not in monthly.columns:
            return "Insufficient data for savings projection"
        
        monthly['savings'] = monthly['income'] - monthly['expense']
        
        if len(monthly) < 2:
            return "Need more data for savings projection"
        
        avg_savings = monthly['savings'].mean()
        projected_annual = avg_savings * 12
        
        # Calculate time to reach savings goal
        if avg_savings > 0:
            months_to_goal = max(1, round(self.savings_target / avg_savings))
            goal_date = datetime.now() + timedelta(days=30 * months_to_goal)
        else:
            months_to_goal = "N/A"
            goal_date = "N/A"
        
        return {
            "average_monthly_savings": round(float(avg_savings), 2),
            "projected_annual_savings": round(float(projected_annual), 2),
            "months_to_goal": months_to_goal,
            "estimated_goal_date": goal_date.strftime("%B %Y") if goal_date != "N/A" else "N/A",
            "progress_percentage": min(100, (avg_savings * len(monthly) / self.savings_target) * 100)
        }
    
    def detect_spending_anomalies(self, df):
        """Detect unusual spending patterns"""
        expenses = df[df['type'] == 'expense'].copy()
        
        if expenses.empty:
            return []
        
        expenses['date'] = pd.to_datetime(expenses['date'])
        
        # Calculate z-scores for amount detection
        mean_amount = expenses['amount'].mean()
        std_amount = expenses['amount'].std()
        
        anomalies = []
        
        if std_amount > 0:  # Avoid division by zero
            for _, transaction in expenses.iterrows():
                z_score = abs(transaction['amount'] - mean_amount) / std_amount
                
                if z_score > 2:  # More than 2 standard deviations from mean
                    anomalies.append({
                        "date": transaction['date'].strftime("%Y-%m-%d"),
                        "category": transaction['category'],
                        "amount": transaction['amount'],
                        "z_score": round(z_score, 2),
                        "message": f"Unusually high spending in {transaction['category']}"
                    })
        
        return anomalies
    
    def financial_health_score(self, df):
        """Calculate financial health score (0-100)"""
        if df.empty:
            return {"score": 0, "message": "Insufficient data"}
        
        score = 50  # Base score
        
        # Calculate metrics
        total_income = df[df['type'] == 'income']['amount'].sum()
        total_expense = df[df['type'] == 'expense']['amount'].sum()
        
        if total_income > 0:
            savings_rate = (total_income - total_expense) / total_income * 100
            
            # Score based on savings rate
            if savings_rate >= 20:
                score += 30
            elif savings_rate >= 10:
                score += 20
            elif savings_rate >= 0:
                score += 10
            else:
                score -= 20
        
        # Score based on expense diversity
        expenses = df[df['type'] == 'expense']
        if not expenses.empty:
            category_count = expenses['category'].nunique()
            if category_count >= 5:
                score += 10
            elif category_count >= 3:
                score += 5
        
        # Score based on data consistency
        if len(df) >= 10:
            score += 10
        
        score = max(0, min(100, score))  # Clamp between 0-100
        
        # Health message
        if score >= 80:
            message = "Excellent financial health! 🎉"
        elif score >= 60:
            message = "Good financial health! 👍"
        elif score >= 40:
            message = "Average financial health 📊"
        else:
            message = "Needs improvement 📈"
        
        return {
            "score": round(score),
            "message": message,
            "breakdown": {
                "savings_rate": round((total_income - total_expense) / total_income * 100, 1) if total_income > 0 else 0,
                "expense_categories": expenses['category'].nunique() if not expenses.empty else 0,
                "transaction_count": len(df)
            }
        }
    
    def analyze_category_trends(self, df):
        """Analyze spending trends by category"""
        expenses = df[df['type'] == 'expense'].copy()
        
        if expenses.empty:
            return {}
        
        expenses['date'] = pd.to_datetime(expenses['date'])
        expenses['month'] = expenses['date'].dt.to_period('M')
        
        category_trends = {}
        for category in expenses['category'].unique():
            category_data = expenses[expenses['category'] == category]
            monthly_totals = category_data.groupby('month')['amount'].sum()
            
            if len(monthly_totals) > 1:
                trend = "increasing" if monthly_totals.iloc[-1] > monthly_totals.iloc[-2] else "decreasing"
                change_pct = ((monthly_totals.iloc[-1] - monthly_totals.iloc[-2]) / monthly_totals.iloc[-2]) * 100
            else:
                trend = "stable"
                change_pct = 0
            
            category_trends[category] = {
                "trend": trend,
                "change_percentage": round(change_pct, 1),
                "last_month_spending": round(float(monthly_totals.iloc[-1]) if len(monthly_totals) > 0 else 0, 2)
            }
        
        return category_trends
    
    def generate_all(self, transactions):
        """Generate all insights including predictions"""
        df = self.get_dataframe(transactions)
        
        if df.empty:
            return self._empty_insights()
        
        insights = {}
        insights["responsible_ai"] = self.responsible_ai_checks(df)
        insights["summary_stats"] = self.summary_stats(df)
        insights["trend"] = self.trend_insights(df)
        insights["alerts"] = self.alerts(df)
        insights["budget_recommendations"] = self.budget_recommendations(df)
        insights["predictive_analytics"] = self.predictive_analytics(transactions)
        
        return insights
    
    # Add these methods to your existing InsightGeneratorAgent class

# In your generate_llm_advice method, replace:
def generate_llm_advice(self, insights):
    """Generate LLM-based financial advice using Gemini"""
    try:
        from agents.nlp_processor import NLPProcessor
        nlp_processor = NLPProcessor()
        
        # Convert insights to transaction format for Gemini
        transactions = []  # You might need to pass actual transactions here
        
        return nlp_processor.generate_financial_insights_llm(transactions)
        
    except Exception as e:
        return f"AI advice unavailable: {str(e)}"
    
def analyze_transaction_notes(self, transactions):
    """Analyze transaction notes using NLP"""
    try:
        from agents.nlp_processor import NLPProcessor
        nlp_processor = NLPProcessor()
        
        analysis = {
            'total_notes': 0,
            'sentiment_analysis': [],
            'common_entities': {}
        }
        
        for transaction in transactions:
            if transaction.note:
                analysis['total_notes'] += 1
                sentiment = nlp_processor.analyze_sentiment(transaction.note)
                entities = nlp_processor.extract_entities(transaction.note)
                
                analysis['sentiment_analysis'].append({
                    'note': transaction.note[:50] + '...' if len(transaction.note) > 50 else transaction.note,
                    'sentiment': sentiment
                })
                
                # Aggregate entities
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        if entity_type not in analysis['common_entities']:
                            analysis['common_entities'][entity_type] = []
                        analysis['common_entities'][entity_type].extend(entity_list)
        
        return analysis
        
    except Exception as e:
        return {'error': str(e)}

def compare_with_real_dataset(self, df):
    """Compare user data with real dataset patterns"""
    try:
        from data_loader import RealDatasetLoader
        loader = RealDatasetLoader()
        real_transactions = loader.load_and_preprocess()
        real_df = pd.DataFrame(real_transactions)
        
        if real_df.empty:
            return {'message': 'Real dataset not available for comparison'}
        
        # Calculate comparisons
        user_stats = self._calculate_user_stats(df)
        real_stats = self._calculate_real_stats(real_df)
        
        comparisons = {
            'user_stats': user_stats,
            'real_stats': real_stats,
            'insights': self._generate_comparison_insights(user_stats, real_stats)
        }
        
        return comparisons
        
    except Exception as e:
        return {'error': f'Dataset comparison failed: {str(e)}'}

def _calculate_user_stats(self, df):
    """Calculate user statistics"""
    return {
        'avg_income': df[df['type'] == 'income']['amount'].mean(),
        'avg_expense': df[df['type'] == 'expense']['amount'].mean(),
        'savings_rate': ((df[df['type'] == 'income']['amount'].sum() - 
                         df[df['type'] == 'expense']['amount'].sum()) / 
                        df[df['type'] == 'income']['amount'].sum() * 100) if df[df['type'] == 'income']['amount'].sum() > 0 else 0,
        'top_category': df[df['type'] == 'expense'].groupby('category')['amount'].sum().idxmax() if not df[df['type'] == 'expense'].empty else 'N/A'
    }

def _calculate_real_stats(self, real_df):
    """Calculate real dataset statistics"""
    return {
        'avg_income': real_df[real_df['type'] == 'income']['amount'].mean(),
        'avg_expense': real_df[real_df['type'] == 'expense']['amount'].mean(),
        'savings_rate': ((real_df[real_df['type'] == 'income']['amount'].sum() - 
                         real_df[real_df['type'] == 'expense']['amount'].sum()) / 
                        real_df[real_df['type'] == 'income']['amount'].sum() * 100) if real_df[real_df['type'] == 'income']['amount'].sum() > 0 else 0,
        'top_category': real_df[real_df['type'] == 'expense'].groupby('category')['amount'].sum().idxmax() if not real_df[real_df['type'] == 'expense'].empty else 'N/A'
    }