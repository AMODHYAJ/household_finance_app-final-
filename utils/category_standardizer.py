# utils/category_standardizer.py
import re
from typing import Dict, List

class CategoryStandardizer:
    """Standardize category names to ensure consistency"""
    
    def __init__(self):
        # Define standard categories and their variations
        self.standard_categories = {
            'Food': ['food', 'groceries', 'grocery', 'restaurant', 'dining', 'meal', 
                    'supermarket', 'cafe', 'coffee', 'pizza', 'burger', 'lunch', 
                    'dinner', 'breakfast', 'eating', 'ate', 'bakery', 'starbucks'],
            'Transport': ['transport', 'transportation', 'uber', 'taxi', 'bus', 'train', 
                         'flight', 'cab', 'lyft', 'gas', 'fuel', 'petrol', 'commute', 
                         'parking', 'airport'],
            'Shopping': ['shopping', 'store', 'mall', 'retail', 'purchase', 'buy', 
                        'amazon', 'ebay', 'walmart', 'target', 'clothes', 'shoes', 
                        'book', 'gift'],
            'Entertainment': ['entertainment', 'movie', 'netflix', 'spotify', 'game', 
                            'gaming', 'concert', 'cinema', 'theater', 'fun', 'hobby', 
                            'leisure', 'recreation'],
            'Bills': ['bills', 'bill', 'utility', 'electric', 'electricity', 'water', 
                     'internet', 'phone', 'mobile', 'wifi', 'rent', 'mortgage', 
                     'subscription'],
            'Healthcare': ['healthcare', 'health', 'medical', 'doctor', 'hospital', 
                          'pharmacy', 'insurance', 'clinic', 'medicine'],
            'Education': ['education', 'school', 'university', 'course', 'book', 
                         'tuition', 'college', 'student'],
            'Income': ['income', 'salary', 'paycheck', 'payment', 'wage', 'earnings', 
                      'bonus', 'freelance', 'invoice', 'stipend', 'refund'],
            'Other': ['other', 'misc', 'miscellaneous', 'general']
        }
        
        # Create reverse mapping for quick lookup
        self.variation_to_standard = {}
        for standard, variations in self.standard_categories.items():
            for variation in variations:
                self.variation_to_standard[variation.lower()] = standard
    
    def standardize(self, category: str) -> str:
        """Convert any category variation to standard form"""
        if not category or not isinstance(category, str):
            return 'Other'
        
        category_lower = category.strip().lower()
        
        # Direct match
        if category_lower in self.variation_to_standard:
            return self.variation_to_standard[category_lower]
        
        # Partial match (if category contains a keyword)
        for variation, standard in self.variation_to_standard.items():
            if variation in category_lower:
                return standard
        
        # Title case if it looks like a proper category but not in our list
        if category_lower and any(c.isalpha() for c in category):
            return category.title()
        
        return 'Other'
    
    def get_standard_categories(self) -> List[str]:
        """Get list of all standard categories"""
        return list(self.standard_categories.keys())
    
    def is_valid_category(self, category: str) -> bool:
        """Check if category is valid (can be standardized)"""
        standardized = self.standardize(category)
        return standardized != 'Other' or category.title() == 'Other'

# Global instance
category_standardizer = CategoryStandardizer()