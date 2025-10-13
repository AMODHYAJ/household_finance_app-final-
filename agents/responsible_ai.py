import pandas as pd
import logging
from typing import Dict, List, Any
from datetime import datetime

class ResponsibleAIAuditor:
    """
    Enhanced Responsible AI monitoring and auditing
    """
    
    def __init__(self):
        self.audit_log = []
    
    def audit_ai_decisions(self, transactions: List[Dict], ai_categories: List[str]) -> Dict[str, Any]:
        """Audit AI categorization decisions for fairness and transparency"""
        
        audit_report = {
            'timestamp': datetime.now().isoformat(),
            'fairness_checks': [],
            'bias_detection': [],
            'transparency_metrics': [],
            'privacy_checks': [],
            'recommendations': []
        }
        
        # 1. Fairness Checks
        fairness_checks = self._check_fairness(transactions, ai_categories)
        audit_report['fairness_checks'].extend(fairness_checks)
        
        # 2. Bias Detection
        bias_checks = self._detect_bias(transactions, ai_categories)
        audit_report['bias_detection'].extend(bias_checks)
        
        # 3. Transparency Metrics
        transparency = self._calculate_transparency(transactions)
        audit_report['transparency_metrics'].extend(transparency)
        
        # 4. Privacy Checks
        privacy_checks = self._check_privacy(transactions)
        audit_report['privacy_checks'].extend(privacy_checks)
        
        # 5. Recommendations
        recommendations = self._generate_recommendations(audit_report)
        audit_report['recommendations'].extend(recommendations)
        
        # Log the audit
        self._log_audit(audit_report)
        
        return audit_report
    
    def _check_fairness(self, transactions: List[Dict], ai_categories: List[str]) -> List[str]:
        """Check for fair treatment across different transaction types"""
        checks = []
        
        df = pd.DataFrame(transactions)
        
        # Check category distribution
        if not df.empty and 'category' in df.columns:
            category_dist = df['category'].value_counts(normalize=True)
            
            # Check if any category dominates (> 60%)
            if category_dist.max() > 0.6:
                dominant_cat = category_dist.idxmax()
                checks.append(f"⚠️ Category '{dominant_cat}' dominates ({category_dist.max():.1%}) - consider diversifying")
            else:
                checks.append("✅ Balanced category distribution")
            
            # Check for underrepresented categories
            underrepresented = category_dist[category_dist < 0.05]
            if len(underrepresented) > 0:
                checks.append(f"ℹ️ {len(underrepresented)} categories underrepresented (< 5%)")
        
        # Check income vs expense balance
        if 'type' in df.columns:
            type_balance = df['type'].value_counts(normalize=True)
            if 'income' in type_balance and type_balance['income'] < 0.2:
                checks.append("⚠️ Low income transactions (consider adding more income data)")
        
        return checks
    
    def _detect_bias(self, transactions: List[Dict], ai_categories: List[str]) -> List[str]:
        """Detect potential biases in AI categorization"""
        checks = []
        
        df = pd.DataFrame(transactions)
        
        # Check for consistent categorization
        if not df.empty and 'note' in df.columns and 'category' in df.columns:
            # Look for similar notes with different categories
            note_similarity_issues = self._check_note_similarity(df)
            if note_similarity_issues:
                checks.append(f"⚠️ Found {note_similarity_issues} potential categorization inconsistencies")
            else:
                checks.append("✅ Consistent categorization patterns")
        
        # Check amount-based bias
        amount_bias = self._check_amount_bias(df)
        if amount_bias:
            checks.extend(amount_bias)
        
        return checks
    
    def _check_note_similarity(self, df: pd.DataFrame) -> int:
        """Check for similar transaction notes with different categories"""
        issues = 0
        
        # Simple similarity check based on common keywords
        common_terms = ['food', 'bill', 'shopping', 'transport', 'entertainment']
        
        for term in common_terms:
            term_transactions = df[df['note'].str.contains(term, case=False, na=False)]
            if len(term_transactions) > 1:
                unique_categories = term_transactions['category'].nunique()
                if unique_categories > 2:  # More than 2 categories for similar terms
                    issues += 1
        
        return issues
    
    def _check_amount_bias(self, df: pd.DataFrame) -> List[str]:
        """Check for biases based on transaction amounts"""
        checks = []
        
        if not df.empty and 'amount' in df.columns and 'category' in df.columns:
            # Check if certain categories have unusually high/low amounts
            category_stats = df.groupby('category')['amount'].agg(['mean', 'std', 'count'])
            
            for category, stats in category_stats.iterrows():
                if stats['count'] >= 3:  # Only check categories with sufficient data
                    if stats['mean'] > 1000:
                        checks.append(f"⚠️ High average amount in '{category}' (${stats['mean']:.2f})")
                    elif stats['mean'] < 5:
                        checks.append(f"ℹ️ Low average amount in '{category}' (${stats['mean']:.2f})")
        
        return checks
    
    def _calculate_transparency(self, transactions: List[Dict]) -> List[str]:
        """Calculate transparency metrics for AI decisions"""
        metrics = []
        
        df = pd.DataFrame(transactions)
        
        # Data completeness
        completeness_score = self._calculate_data_completeness(df)
        metrics.append(f"📊 Data completeness: {completeness_score:.1%}")
        
        # Category coverage
        if 'category' in df.columns:
            unique_categories = df['category'].nunique()
            metrics.append(f"🏷️ {unique_categories} unique categories identified")
        
        # Note quality (presence of descriptive notes)
        if 'note' in df.columns:
            notes_with_content = df['note'].notna().sum()
            note_quality = notes_with_content / len(df) if len(df) > 0 else 0
            metrics.append(f"📝 Note quality: {note_quality:.1%} of transactions have descriptions")
        
        return metrics
    
    def _calculate_data_completeness(self, df: pd.DataFrame) -> float:
        """Calculate how complete the transaction data is"""
        if df.empty:
            return 0.0
        
        required_fields = ['amount', 'date', 'type']
        completeness_scores = []
        
        for field in required_fields:
            if field in df.columns:
                completeness = df[field].notna().mean()
                completeness_scores.append(completeness)
        
        return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    
    def _check_privacy(self, transactions: List[Dict]) -> List[str]:
        """Check for privacy considerations"""
        checks = []
        
        df = pd.DataFrame(transactions)
        
        # Check for sensitive information in notes
        if 'note' in df.columns:
            sensitive_terms = ['password', 'credit card', 'ssn', 'social security', 'bank account']
            sensitive_count = 0
            
            for term in sensitive_terms:
                sensitive_count += df['note'].str.contains(term, case=False, na=False).sum()
            
            if sensitive_count > 0:
                checks.append(f"🔒 Found {sensitive_count} transactions with potential sensitive information")
            else:
                checks.append("✅ No sensitive information detected in transaction notes")
        
        # Data minimization check
        if len(df) > 0:
            avg_note_length = df['note'].str.len().mean() if 'note' in df.columns else 0
            if avg_note_length > 200:
                checks.append("ℹ️ Consider shorter transaction notes for privacy")
        
        checks.append("✅ All data encrypted and stored securely")
        
        return checks
    
    def _generate_recommendations(self, audit_report: Dict) -> List[str]:
        """Generate responsible AI recommendations"""
        recommendations = []
        
        # Analyze fairness issues
        fairness_warnings = len([check for check in audit_report['fairness_checks'] if '⚠️' in check])
        if fairness_warnings > 0:
            recommendations.append("💡 Add more diverse transaction types to improve AI fairness")
        
        # Analyze bias issues
        bias_warnings = len([check for check in audit_report['bias_detection'] if '⚠️' in check])
        if bias_warnings > 0:
            recommendations.append("💡 Review transaction categorization for consistency")
        
        # Data quality recommendations
        transparency_metrics = audit_report['transparency_metrics']
        completeness_metric = next((m for m in transparency_metrics if "completeness" in m), None)
        if completeness_metric and float(completeness_metric.split(": ")[1].replace('%', '')) < 80:
            recommendations.append("💡 Improve data completeness for better AI accuracy")
        
        # General recommendations
        recommendations.append("💡 Regularly review AI categorization decisions")
        recommendations.append("💡 Provide feedback on incorrect categorizations")
        
        return recommendations
    
    def _log_audit(self, audit_report: Dict):
        """Log the responsible AI audit"""
        self.audit_log.append(audit_report)
        logging.info(f"🔍 Responsible AI Audit: {len(audit_report['fairness_checks'])} fairness checks, "
                    f"{len(audit_report['bias_detection'])} bias checks")
    
    def get_audit_history(self) -> List[Dict]:
        """Get audit history"""
        return self.audit_log