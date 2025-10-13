import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any

class RealDatasetLoader:
    """
    Enhanced real dataset loader with better integration
    """
    
    def __init__(self):
        self.dataset_path = os.getenv('REAL_DATASET_PATH', 'data/raw/household_income_dataset.csv')
        self.categories = ['food', 'housing', 'transport', 'healthcare', 'education', 'entertainment', 'shopping', 'bills']
        self.use_real_data = os.getenv('USE_REAL_DATA', 'True').lower() == 'true'
    
    def load_and_preprocess(self) -> List[Dict[str, Any]]:
        """Load and preprocess the real household dataset"""
        if not self.use_real_data:
            print("⚠️ Real dataset usage disabled in settings")
            return self.generate_sample_data()
            
        try:
            # Check if dataset exists
            if not os.path.exists(self.dataset_path):
                print(f"❌ Dataset not found at: {self.dataset_path}")
                print("🔄 Generating sample dataset...")
                self._create_sample_dataset()
            
            # Load the dataset
            df = pd.read_csv(self.dataset_path)
            print(f"✅ Loaded real dataset with {len(df)} rows")
            
            # Basic preprocessing
            df = self.clean_dataset(df)
            
            # Convert to our transaction format
            transactions = self.convert_to_transactions(df)
            
            print(f"✅ Processed {len(transactions)} transactions from real dataset")
            return transactions
            
        except Exception as e:
            print(f"❌ Error loading real dataset: {e}")
            print("🔄 Falling back to generated data...")
            return self.generate_sample_data()
    
    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset"""
        # Create a copy to avoid warnings
        df_clean = df.copy()
        
        # Handle missing values
        df_clean = df_clean.ffill().bfill()
        
        # Ensure essential columns exist or create them
        if 'amount' not in df_clean.columns:
            if 'income' in df_clean.columns:
                df_clean['amount'] = df_clean['income']
            elif 'expense' in df_clean.columns:
                df_clean['amount'] = df_clean['expense']
            else:
                df_clean['amount'] = np.random.uniform(10, 500, len(df_clean))
        
        if 'category' not in df_clean.columns:
            df_clean['category'] = [random.choice(self.categories) for _ in range(len(df_clean))]
        
        if 'type' not in df_clean.columns:
            # Assume rows with high amounts are income, others are expenses
            df_clean['type'] = df_clean['amount'].apply(
                lambda x: 'income' if x > 1000 else 'expense'
            )
        
        if 'date' not in df_clean.columns:
            start_date = datetime(2024, 1, 1)
            df_clean['date'] = [start_date + timedelta(days=i) for i in range(len(df_clean))]
        else:
            # Convert date strings to datetime
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        return df_clean
    
    def convert_to_transactions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert dataset to transaction format used in the app"""
        transactions = []
        
        for _, row in df.iterrows():
            transaction_date = row['date'] if 'date' in row else datetime.now()
            
            # Create transaction based on type
            transaction = {
                'date': transaction_date,
                'type': row.get('type', 'expense'),
                'category': row.get('category', 'other'),
                'amount': float(row['amount']),
                'note': self._generate_note(row),
                'source': 'real_dataset'
            }
            
            transactions.append(transaction)
        
        return transactions
    
    def _generate_note(self, row: pd.Series) -> str:
        """Generate descriptive note based on row data"""
        category = row.get('category', 'general')
        amount = row.get('amount', 0)
        
        notes = {
            'food': [f"Groceries ${amount}", f"Dining out ${amount}", f"Supermarket ${amount}"],
            'housing': [f"Rent ${amount}", f"Mortgage ${amount}", f"Home maintenance ${amount}"],
            'transport': [f"Gas ${amount}", f"Public transport ${amount}", f"Car maintenance ${amount}"],
            'entertainment': [f"Movies ${amount}", f"Gaming ${amount}", f"Streaming ${amount}"],
        }
        
        if category in notes:
            return random.choice(notes[category])
        else:
            return f"{category.title()} expense ${amount}"
    
    def generate_sample_data(self) -> List[Dict[str, Any]]:
        """Generate realistic sample data when real dataset is unavailable"""
        print("🔄 Generating realistic sample data...")
        transactions = []
        start_date = datetime(2024, 1, 1)
        
        # Realistic household patterns
        household_patterns = [
            {'income_freq': 'monthly', 'income_amount': 4500, 'expense_range': (3000, 4000), 'size': 4},
            {'income_freq': 'biweekly', 'income_amount': 3200, 'expense_range': (2500, 3200), 'size': 3},
            {'income_freq': 'monthly', 'income_amount': 3800, 'expense_range': (2800, 3600), 'size': 2},
        ]
        
        for i in range(90):  # 3 months of data
            current_date = start_date + timedelta(days=i)
            household = random.choice(household_patterns)
            
            # Add income
            if household['income_freq'] == 'monthly' and current_date.day == 1:
                transactions.append({
                    'date': current_date,
                    'type': 'income',
                    'category': 'salary',
                    'amount': household['income_amount'],
                    'note': f"Monthly salary - Household of {household['size']}",
                    'source': 'generated'
                })
            elif household['income_freq'] == 'biweekly' and current_date.day in [1, 15]:
                transactions.append({
                    'date': current_date,
                    'type': 'income',
                    'category': 'salary',
                    'amount': household['income_amount'],
                    'note': f"Biweekly salary - Household of {household['size']}",
                    'source': 'generated'
                })
            
            # Add daily expenses (2-5 per day)
            for _ in range(random.randint(2, 5)):
                category = random.choice(self.categories)
                amount = self._get_realistic_amount(category, household['size'])
                
                transactions.append({
                    'date': current_date,
                    'type': 'expense',
                    'category': category,
                    'amount': amount,
                    'note': self._generate_note({'category': category, 'amount': amount}),
                    'source': 'generated'
                })
        
        print(f"✅ Generated {len(transactions)} sample transactions")
        return transactions
    
    def _get_realistic_amount(self, category: str, household_size: int) -> float:
        """Get realistic amounts based on category and household size"""
        base_amounts = {
            'food': (50, 200),
            'housing': (800, 2000),
            'transport': (20, 100),
            'healthcare': (30, 150),
            'education': (50, 300),
            'entertainment': (20, 80),
            'shopping': (25, 120),
            'bills': (100, 400)
        }
        
        min_amt, max_amt = base_amounts.get(category, (10, 100))
        # Adjust for household size
        adjusted_min = min_amt * (household_size / 2)
        adjusted_max = max_amt * (household_size / 2)
        
        return round(random.uniform(adjusted_min, adjusted_max), 2)
    
    def _create_sample_dataset(self):
        """Create a sample dataset file for demonstration"""
        print("📁 Creating sample dataset...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        
        # Generate sample data
        sample_data = []
        start_date = datetime(2024, 1, 1)
        
        for i in range(100):
            row = {
                'date': (start_date + timedelta(days=i)).strftime('%Y-%m-%d'),
                'income': 5000 if i % 30 == 0 else 0,  # Monthly income
                'expense': random.randint(50, 300),
                'category': random.choice(self.categories),
                'household_size': random.randint(2, 5),
                'region': random.choice(['urban', 'suburban', 'rural'])
            }
            sample_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(sample_data)
        df.to_csv(self.dataset_path, index=False)
        print(f"✅ Created sample dataset at: {self.dataset_path}")
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset"""
        transactions = self.load_and_preprocess()
        df = pd.DataFrame(transactions)
        
        if df.empty:
            return {'error': 'No data available'}
        
        stats = {
            'total_transactions': len(transactions),
            'income_transactions': len(df[df['type'] == 'income']),
            'expense_transactions': len(df[df['type'] == 'expense']),
            'total_income': df[df['type'] == 'income']['amount'].sum(),
            'total_expenses': df[df['type'] == 'expense']['amount'].sum(),
            'categories': df['category'].value_counts().to_dict(),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A',
                'end': df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'
            }
        }
        
        return stats