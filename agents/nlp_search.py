import pandas as pd
import re
from typing import List, Dict, Any
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse

class SemanticSearchEngine:
    """Enhanced NLP-powered transaction search with better understanding"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, min_df=1, max_df=0.8)
        self.transaction_vectors = None
        self.transaction_data = []
        
        # Enhanced keyword mappings
        self.category_synonyms = {
            'food': ['food', 'restaurant', 'dining', 'meal', 'lunch', 'dinner', 'breakfast', 'cafe', 'groceries', 'eat', 'eating', 'pizza', 'burger', 'coffee', 'restaurants', 'food & dining'],
            'shopping': ['shopping', 'store', 'mall', 'amazon', 'walmart', 'purchase', 'buy', 'bought', 'shop', 'retail', 'clothing', 'electronics'],
            'entertainment': ['entertainment', 'movie', 'cinema', 'netflix', 'game', 'games', 'fun', 'hobby', 'streaming', 'subscription', 'music', 'sports'],
            'transport': ['transport', 'transportation', 'uber', 'taxi', 'bus', 'train', 'gas', 'fuel', 'ride', 'travel', 'transit', 'commute', 'flight'],
            'bills': ['bills', 'bill', 'utilities', 'electricity', 'water', 'internet', 'phone', 'utility', 'subscription', 'service', 'payment'],
            'healthcare': ['healthcare', 'medical', 'doctor', 'hospital', 'pharmacy', 'medicine', 'health', 'dental', 'insurance'],
            'education': ['education', 'school', 'university', 'course', 'book', 'books', 'learning', 'tuition', 'student'],
            'other': ['other', 'miscellaneous', 'misc', 'general', 'various', 'personal', 'uncategorized']
        }
        
        self.type_synonyms = {
            'income': ['income', 'salary', 'pay', 'payment', 'earnings', 'revenue', 'money in', 'deposit', 'credit', 'refund'],
            'expense': ['expense', 'spending', 'cost', 'payment', 'bill', 'purchase', 'money out', 'debit', 'charge', 'withdrawal']
        }

    def index_transactions(self, transactions: List[Dict]):
        """Index transactions for semantic search"""
        if not transactions:
            self.transaction_vectors = None
            self.transaction_data = []
            return
        
        self.transaction_data = transactions
        
        # Create enhanced searchable text for each transaction
        search_texts = []
        for transaction in transactions:
            text_parts = []
            
            # Add note (most important)
            if transaction.get('note'):
                text_parts.append(transaction['note'])
            
            # Add category with expanded context
            if transaction.get('category'):
                category = transaction['category'].lower()
                text_parts.append(f"category_{category}")
                # Add category synonyms for better matching
                for main_cat, synonyms in self.category_synonyms.items():
                    if self._category_matches(category, main_cat):
                        text_parts.extend(synonyms)
                        break
            
            # Add type with expanded context
            if transaction.get('type'):
                t_type = transaction['type'].lower()
                text_parts.append(f"type_{t_type}")
                # Add type synonyms for better matching
                for main_type, synonyms in self.type_synonyms.items():
                    if self._type_matches(t_type, main_type):
                        text_parts.extend(synonyms)
                        break
            
            # Add amount context with more granular ranges
            if transaction.get('amount'):
                amount = transaction['amount']
                if amount > 500:
                    text_parts.append("very_large_amount expensive premium high_cost luxury")
                elif amount > 100:
                    text_parts.append("large_amount expensive high_cost significant")
                elif amount > 50:
                    text_parts.append("medium_amount moderate_cost average")
                elif amount > 10:
                    text_parts.append("small_amount cheap low_cost inexpensive")
                else:
                    text_parts.append("very_small_amount cheap low_cost minimal")
                
                # Add exact amount as text for pattern matching
                text_parts.append(f"amount_{int(amount)}")
            
            # Add date context for time-based queries
            if transaction.get('date'):
                try:
                    if isinstance(transaction['date'], str):
                        trans_date = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00'))
                    else:
                        trans_date = transaction['date']
                    
                    today = datetime.now().date()
                    if trans_date.date() == today:
                        text_parts.extend(["today", "recent", "current_day"])
                    elif trans_date.date() == today - timedelta(days=1):
                        text_parts.extend(["yesterday", "recent", "last_day"])
                    elif trans_date.date() >= today - timedelta(days=7):
                        text_parts.extend(["this_week", "recent", "last_7_days"])
                    elif trans_date.date() >= today - timedelta(days=30):
                        text_parts.extend(["this_month", "recent", "last_30_days"])
                except:
                    pass
            
            search_text = " ".join(text_parts)
            search_texts.append(search_text)
        
        # Create TF-IDF vectors
        if search_texts:
            try:
                self.transaction_vectors = self.vectorizer.fit_transform(search_texts)
                print(f"✅ Indexed {len(search_texts)} transactions with vocabulary size: {len(self.vectorizer.vocabulary_)}")
            except Exception as e:
                print(f"Error creating TF-IDF vectors: {e}")
                self.transaction_vectors = None
        else:
            self.transaction_vectors = None

    def semantic_search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Find transactions semantically similar to query"""
        if self.transaction_vectors is None or not self.transaction_data:
            return []
        
        # Enhance query with synonyms and context
        enhanced_query = self._enhance_query(query)
        
        try:
            # Transform query to same vector space
            query_vector = self.vectorizer.transform([enhanced_query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vector, self.transaction_vectors).flatten()
            
            # Get top matches with lower threshold for better recall
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Lower threshold for better recall
                    transaction = self.transaction_data[idx].copy()
                    transaction['similarity_score'] = round(similarities[idx], 3)
                    transaction['match_reason'] = self._explain_match(transaction, query)
                    results.append(transaction)
            
            return results
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    def _enhance_query(self, query: str) -> str:
        """Enhance query with synonyms and contextual understanding"""
        query_lower = query.lower()
        enhanced_terms = [query_lower]
        
        # Add category synonyms
        for category, synonyms in self.category_synonyms.items():
            if any(term in query_lower for term in synonyms):
                enhanced_terms.extend(synonyms)
        
        # Add type synonyms
        for t_type, synonyms in self.type_synonyms.items():
            if any(term in query_lower for term in synonyms):
                enhanced_terms.extend(synonyms)
        
        # Add time context
        time_terms = ['today', 'yesterday', 'week', 'month', 'recent', 'last']
        if any(term in query_lower for term in time_terms):
            enhanced_terms.extend(['today', 'yesterday', 'this_week', 'last_week', 'this_month', 'last_month', 'recent'])
        
        # Add amount context
        amount_patterns = [
            (r'\$?(\d+)', 'exact'),
            (r'over\s+\$?(\d+)', 'over'),
            (r'more than\s+\$?(\d+)', 'over'),
            (r'under\s+\$?(\d+)', 'under'),
            (r'less than\s+\$?(\d+)', 'under'),
            (r'above\s+\$?(\d+)', 'over'),
            (r'below\s+\$?(\d+)', 'under')
        ]
        
        for pattern, op in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = float(match.group(1))
                enhanced_terms.extend(['amount', 'money', 'cost', 'price', 'transaction_amount'])
                # Add amount range context
                if op == 'over':
                    enhanced_terms.extend(['large_amount', 'expensive', 'high_cost', 'significant'])
                elif op == 'under':
                    enhanced_terms.extend(['small_amount', 'cheap', 'low_cost', 'inexpensive'])
                break
        
        # Add size context
        if any(term in query_lower for term in ['large', 'big', 'huge', 'significant']):
            enhanced_terms.extend(['large_amount', 'expensive', 'high_cost', 'significant'])
        if any(term in query_lower for term in ['small', 'little', 'minor']):
            enhanced_terms.extend(['small_amount', 'cheap', 'low_cost', 'inexpensive'])
        
        return " ".join(set(enhanced_terms))  # Remove duplicates

    def _explain_match(self, transaction: Dict, query: str) -> str:
        """Generate explanation for why transaction matches query"""
        reasons = []
        query_lower = query.lower()
        
        # Check category match
        if transaction.get('category'):
            category = transaction['category'].lower()
            for main_cat, synonyms in self.category_synonyms.items():
                if self._category_matches(category, main_cat) and any(term in query_lower for term in synonyms):
                    reasons.append(f"category: {transaction['category']}")
                    break
        
        # Check type match
        if transaction.get('type'):
            t_type = transaction['type'].lower()
            for main_type, synonyms in self.type_synonyms.items():
                if self._type_matches(t_type, main_type) and any(term in query_lower for term in synonyms):
                    reasons.append(f"type: {transaction['type']}")
                    break
        
        # Check note match
        if transaction.get('note'):
            note_lower = transaction['note'].lower()
            query_words = query_lower.split()
            matched_words = [word for word in query_words if len(word) > 2 and word in note_lower]
            if matched_words:
                reasons.append(f"note contains: {', '.join(matched_words)}")
        
        # Check amount match
        if transaction.get('amount'):
            amount = transaction['amount']
            
            # Check for "over X" patterns
            over_match = re.search(r'over\s+\$?(\d+)', query_lower)
            if over_match:
                value = float(over_match.group(1))
                if amount > value:
                    reasons.append(f"amount over ${value}")
            
            # Check for "under X" patterns
            under_match = re.search(r'under\s+\$?(\d+)', query_lower)
            if under_match:
                value = float(under_match.group(1))
                if amount < value:
                    reasons.append(f"amount under ${value}")
            
            # Check for exact amount mentions
            exact_match = re.search(r'\$?(\d+)', query_lower)
            if exact_match and not over_match and not under_match:
                value = float(exact_match.group(1))
                if abs(amount - value) <= value * 0.3:  # 30% tolerance
                    reasons.append(f"amount around ${value}")
        
        return ", ".join(reasons) if reasons else "semantic match"

    def natural_language_search(self, query: str, transactions: List[Dict]) -> List[Dict]:
        """Main search method that combines semantic and structured search"""
        if not transactions:
            return []
        
        print(f"🔍 Processing query: '{query}'")
        
        # Parse query for structured filters first
        parsed_query = self._parse_natural_language(query)
        print(f"📝 Parsed filters: {parsed_query['filters']}")
        
        # Apply structured filters first (more precise)
        filtered_transactions = self._apply_structured_filters(transactions, parsed_query)
        print(f"🎯 After structured filtering: {len(filtered_transactions)} transactions")
        
        # If we have good structured matches, use those
        if filtered_transactions and len(filtered_transactions) <= 20:
            print("✅ Using structured filter results")
            results = filtered_transactions
            # Add scoring for structured matches
            for result in results:
                result['similarity_score'] = 0.8  # High score for direct matches
                result['match_reason'] = self._explain_structured_match(result, parsed_query)
        else:
            # Use semantic search
            print("🤖 Using semantic search")
            self.index_transactions(transactions)
            
            if self.transaction_vectors is not None:
                results = self.semantic_search(query, top_k=20)
            else:
                results = []
            
            # If semantic search returns few results, try with filtered set
            if len(results) < 5 and filtered_transactions:
                print("🔄 Trying semantic search on filtered transactions")
                self.index_transactions(filtered_transactions)
                if self.transaction_vectors is not None:
                    semantic_results = self.semantic_search(query, top_k=20)
                    results.extend(semantic_results)
        
        # Remove duplicates and sort
        seen_ids = set()
        unique_results = []
        for result in results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                unique_results.append(result)
        
        # Sort by relevance score and date
        unique_results.sort(key=lambda x: (x.get('similarity_score', 0), x.get('date', '')), reverse=True)
        
        final_results = unique_results[:10]
        print(f"🎯 Final results: {len(final_results)} transactions")
        
        return final_results

    def _explain_structured_match(self, transaction: Dict, parsed_query: Dict) -> str:
        """Explain why transaction matches structured filters"""
        reasons = []
        
        if 'category' in parsed_query['filters']:
            reasons.append(f"category: {transaction.get('category', 'unknown')}")
        
        if 'type' in parsed_query['filters']:
            reasons.append(f"type: {transaction.get('type', 'unknown')}")
        
        if 'amount' in parsed_query['filters']:
            amount_filter = parsed_query['filters']['amount']
            reasons.append(f"amount {amount_filter['operator']} ${amount_filter['value']}")
        
        if parsed_query['time_period']:
            reasons.append(f"time: {parsed_query['time_period']}")
        
        return ", ".join(reasons) if reasons else "structured match"

    def _parse_natural_language(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured filters"""
        query_lower = query.lower()
        parsed = {
            'original_query': query,
            'processed_query': query,
            'filters': {},
            'time_period': None
        }
        
        # Time period detection (improved)
        time_patterns = {
            'today': '1d',
            'yesterday': '1d',
            'this week': '7d',
            'last week': '7d', 
            'this month': '30d',
            'last month': '30d',
            'last 3 months': '90d',
            'recent': '30d',
            'past week': '7d',
            'past month': '30d'
        }
        
        for pattern, period in time_patterns.items():
            if pattern in query_lower:
                parsed['time_period'] = period
                parsed['processed_query'] = parsed['processed_query'].replace(pattern, '')
                break
        
        # Amount filters (improved patterns)
        amount_patterns = [
            (r'over\s+\$?(\d+(?:\.\d+)?)', 'gt'),
            (r'more than\s+\$?(\d+(?:\.\d+)?)', 'gt'),
            (r'above\s+\$?(\d+(?:\.\d+)?)', 'gt'),
            (r'greater than\s+\$?(\d+(?:\.\d+)?)', 'gt'),
            (r'under\s+\$?(\d+(?:\.\d+)?)', 'lt'), 
            (r'less than\s+\$?(\d+(?:\.\d+)?)', 'lt'),
            (r'below\s+\$?(\d+(?:\.\d+)?)', 'lt'),
            (r'up to\s+\$?(\d+(?:\.\d+)?)', 'lt')
        ]
        
        for pattern, op in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = float(match.group(1))
                parsed['filters']['amount'] = {'operator': op, 'value': amount}
                parsed['processed_query'] = re.sub(pattern, '', parsed['processed_query'], flags=re.IGNORECASE)
                break
        
        # Category detection using synonyms (improved)
        for category, synonyms in self.category_synonyms.items():
            if any(synonym in query_lower for synonym in synonyms):
                parsed['filters']['category'] = category
                # Remove matched category terms from processed query
                for synonym in synonyms:
                    if synonym in query_lower:
                        parsed['processed_query'] = parsed['processed_query'].replace(synonym, '')
                break
        
        # Type detection using synonyms (improved)
        for t_type, synonyms in self.type_synonyms.items():
            if any(synonym in query_lower for synonym in synonyms):
                parsed['filters']['type'] = t_type
                for synonym in synonyms:
                    if synonym in query_lower:
                        parsed['processed_query'] = parsed['processed_query'].replace(synonym, '')
                break
        
        # Clean up processed query
        parsed['processed_query'] = re.sub(r'\s+', ' ', parsed['processed_query']).strip()
        
        return parsed

    def _apply_structured_filters(self, transactions: List[Dict], parsed_query: Dict) -> List[Dict]:
        """Apply structured filters to transactions"""
        filtered = transactions
        
        # Time period filter
        if parsed_query['time_period']:
            filtered = self._filter_by_time_period(filtered, parsed_query['time_period'])
        
        # Amount filters
        if 'amount' in parsed_query['filters']:
            amount_filter = parsed_query['filters']['amount']
            filtered = self._filter_by_amount(filtered, amount_filter)
        
        # Category filter
        if 'category' in parsed_query['filters']:
            target_category = parsed_query['filters']['category']
            filtered = [t for t in filtered if self._category_matches(t.get('category', '').lower(), target_category)]
        
        # Type filter  
        if 'type' in parsed_query['filters']:
            target_type = parsed_query['filters']['type']
            filtered = [t for t in filtered if self._type_matches(t.get('type', '').lower(), target_type)]
        
        return filtered

    def _category_matches(self, transaction_category: str, target_category: str) -> bool:
        """Check if transaction category matches target category using synonyms"""
        if transaction_category == target_category:
            return True
        
        # Check synonyms
        synonyms = self.category_synonyms.get(target_category, [])
        return any(syn in transaction_category for syn in synonyms)

    def _type_matches(self, transaction_type: str, target_type: str) -> bool:
        """Check if transaction type matches target type using synonyms"""
        if transaction_type == target_type:
            return True
        
        # Check synonyms
        synonyms = self.type_synonyms.get(target_type, [])
        return any(syn in transaction_type for syn in synonyms)

    def _filter_by_time_period(self, transactions: List[Dict], period: str) -> List[Dict]:
        """Filter transactions by time period"""
        now = datetime.now()
        
        if period == '1d':
            cutoff = now - timedelta(days=1)
        elif period == '7d':
            cutoff = now - timedelta(days=7)
        elif period == '30d':
            cutoff = now - timedelta(days=30)
        elif period == '90d':
            cutoff = now - timedelta(days=90)
        else:
            return transactions
        
        filtered = []
        for transaction in transactions:
            if 'date' in transaction:
                try:
                    if isinstance(transaction['date'], str):
                        trans_date = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00'))
                    else:
                        trans_date = transaction['date']
                    
                    if trans_date >= cutoff:
                        filtered.append(transaction)
                except Exception as e:
                    print(f"Error parsing date: {e}")
                    continue
        
        return filtered

    def _filter_by_amount(self, transactions: List[Dict], amount_filter: Dict) -> List[Dict]:
        """Filter transactions by amount"""
        filtered = []
        value = amount_filter['value']
        operator = amount_filter['operator']
        
        for transaction in transactions:
            if 'amount' not in transaction:
                continue
            
            amount = transaction['amount']
            
            if operator == 'gt' and amount > value:
                filtered.append(transaction)
            elif operator == 'lt' and amount < value:
                filtered.append(transaction)
        
        return filtered