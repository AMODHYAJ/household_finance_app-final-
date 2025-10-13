import spacy
import pandas as pd
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticSearchEngine:
    """Advanced NLP-powered transaction search"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Semantic search engine loaded")
        except OSError:
            print("⚠️ spaCy model not found for semantic search")
            self.nlp = None
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.transaction_vectors = None
        self.transaction_data = []
    
    def index_transactions(self, transactions: List[Dict]):
        """Index transactions for semantic search"""
        if not transactions:
            return
        
        self.transaction_data = transactions
        
        # Create searchable text for each transaction
        search_texts = []
        for transaction in transactions:
            text_parts = []
            
            if transaction.get('note'):
                text_parts.append(transaction['note'])
            if transaction.get('category'):
                text_parts.append(f"category {transaction['category']}")
            if transaction.get('type'):
                text_parts.append(f"type {transaction['type']}")
            
            search_text = " ".join(text_parts)
            search_texts.append(search_text)
        
        # Create TF-IDF vectors
        if search_texts:
            self.transaction_vectors = self.vectorizer.fit_transform(search_texts)
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find transactions semantically similar to query"""
        if not self.transaction_vectors or not self.transaction_data:
            return []
        
        # Transform query to same vector space
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.transaction_vectors).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                transaction = self.transaction_data[idx].copy()
                transaction['similarity_score'] = round(similarities[idx], 3)
                results.append(transaction)
        
        return results
    
    def natural_language_filter(self, query: str, transactions: List[Dict]) -> List[Dict]:
        """Filter transactions using natural language queries"""
        if not transactions:
            return []
        
        self.index_transactions(transactions)
        
        # Enhanced query understanding
        understood_query = self._understand_query_intent(query)
        
        if understood_query['search_type'] == 'semantic':
            return self.semantic_search(understood_query['processed_query'])
        elif understood_query['search_type'] == 'structured':
            return self._structured_filter(understood_query, transactions)
        else:
            return self.semantic_search(query)
    
    def _understand_query_intent(self, query: str) -> Dict[str, Any]:
        """Understand the intent behind natural language queries"""
        query_lower = query.lower()
        intent = {
            'original_query': query,
            'processed_query': query,
            'search_type': 'semantic',
            'filters': {},
            'time_period': None
        }
        
        # Time period detection
        time_patterns = {
            'last week': '7d',
            'last month': '30d', 
            'this month': 'current_month',
            'last 3 months': '90d',
            'recent': '30d'
        }
        
        for pattern, period in time_patterns.items():
            if pattern in query_lower:
                intent['time_period'] = period
                intent['processed_query'] = intent['processed_query'].replace(pattern, '')
        
        # Amount filters
        amount_patterns = [
            (r'over \$?(\d+)', 'gt'),
            (r'more than \$?(\d+)', 'gt'),
            (r'under \$?(\d+)', 'lt'), 
            (r'less than \$?(\d+)', 'lt'),
            (r'around \$?(\d+)', 'approx')
        ]
        
        for pattern, op in amount_patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = float(match.group(1))
                intent['filters']['amount'] = {'operator': op, 'value': amount}
                intent['processed_query'] = re.sub(pattern, '', intent['processed_query'])
        
        # Category detection
        categories = ['food', 'shopping', 'entertainment', 'bills', 'transport', 'healthcare', 'education']
        for category in categories:
            if category in query_lower:
                intent['filters']['category'] = category
                intent['processed_query'] = intent['processed_query'].replace(category, '')
        
        # Type detection
        if 'income' in query_lower:
            intent['filters']['type'] = 'income'
            intent['processed_query'] = intent['processed_query'].replace('income', '')
        elif 'expense' in query_lower:
            intent['filters']['type'] = 'expense' 
            intent['processed_query'] = intent['processed_query'].replace('expense', '')
        
        # Clean up processed query
        intent['processed_query'] = intent['processed_query'].strip()
        
        if intent['filters']:
            intent['search_type'] = 'structured'
        
        return intent
    
    def _structured_filter(self, intent: Dict, transactions: List[Dict]) -> List[Dict]:
        """Apply structured filters based on understood intent"""
        filtered = transactions
        
        # Time period filter
        if intent['time_period']:
            filtered = self._filter_by_time_period(filtered, intent['time_period'])
        
        # Amount filters
        if 'amount' in intent['filters']:
            amount_filter = intent['filters']['amount']
            filtered = self._filter_by_amount(filtered, amount_filter)
        
        # Category filter
        if 'category' in intent['filters']:
            category = intent['filters']['category']
            filtered = [t for t in filtered if t.get('category', '').lower() == category]
        
        # Type filter  
        if 'type' in intent['filters']:
            t_type = intent['filters']['type']
            filtered = [t for t in filtered if t.get('type', '').lower() == t_type]
        
        return filtered
    
    def _filter_by_time_period(self, transactions: List[Dict], period: str) -> List[Dict]:
        """Filter transactions by time period"""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        
        if period == '7d':
            cutoff = now - timedelta(days=7)
        elif period == '30d':
            cutoff = now - timedelta(days=30)
        elif period == '90d':
            cutoff = now - timedelta(days=90)
        elif period == 'current_month':
            cutoff = datetime(now.year, now.month, 1)
        else:
            return transactions
        
        filtered = []
        for transaction in transactions:
            if 'date' in transaction and isinstance(transaction['date'], (datetime, str)):
                try:
                    if isinstance(transaction['date'], str):
                        trans_date = datetime.fromisoformat(transaction['date'].replace('Z', '+00:00'))
                    else:
                        trans_date = transaction['date']
                    
                    if trans_date >= cutoff:
                        filtered.append(transaction)
                except:
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
            elif operator == 'approx' and abs(amount - value) <= value * 0.2:
                filtered.append(transaction)
        
        return filtered