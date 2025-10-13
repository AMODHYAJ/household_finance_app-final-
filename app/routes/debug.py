from flask import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
import os
from datetime import datetime
from core.advanced_security import AdvancedSecurity
from app import db  # Add this import
from core.database import Transaction  # Add this import

debug_bp = Blueprint('debug', __name__)

@debug_bp.route('/debug/features')
@login_required
def feature_debug():
    features = {}
    
    # Check NLP
    try:
        from agents.nlp_processor import NLPProcessor
        nlp = NLPProcessor()
        features['nlp'] = {
            'status': '✅ Working' if nlp.nlp else '❌ No spaCy model',
            'llm_available': nlp.llm_available,
            'model': 'en_core_web_sm' if nlp.nlp else 'None'
        }
    except Exception as e:
        features['nlp'] = {'status': f'❌ Error: {str(e)}'}
    
    # Check Dataset
    try:
        from data_loader import RealDatasetLoader
        loader = RealDatasetLoader()
        transactions = loader.load_and_preprocess()
        features['dataset'] = {
            'status': '✅ Loaded',
            'transaction_count': len(transactions),
            'path': loader.dataset_path
        }
    except Exception as e:
        features['dataset'] = {'status': f'❌ Error: {str(e)}'}
    
    # Check Security
    try:
        security = AdvancedSecurity()
        features['security'] = {
            'status': '✅ Active',
            'encryption': 'Available',
            'jwt': 'Available'
        }
    except Exception as e:
        features['security'] = {'status': f'❌ Error: {str(e)}'}
    
    # Check Environment Variables
    features['environment'] = {
        'OPENAI_API_KEY': '✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing',
        'SECRET_KEY': '✅ Set' if os.getenv('SECRET_KEY') else '❌ Missing',
        'USE_REAL_DATA': os.getenv('USE_REAL_DATA', 'True')
    }
    
    return render_template('feature_debug.html', features=features)

@debug_bp.route('/test-nlp')
@login_required
def test_nlp():
    from agents.nlp_processor import NLPProcessor
    nlp = NLPProcessor()
    
    test_text = "Spent $50 at Walmart for groceries and paid $100 electricity bill on Monday"
    
    results = {
        'entities': nlp.extract_financial_entities(test_text),
        'sentiment': nlp.analyze_sentiment(test_text),
        'category_basic': nlp.categorize_transaction_basic(test_text, 50),
        'category_llm': nlp.categorize_transaction_llm(test_text, 50),
        'llm_available': nlp.llm_available
    }
    
    return jsonify(results)

@debug_bp.route('/test-dataset')
@login_required
def test_dataset():
    try:
        from data_loader import RealDatasetLoader
        loader = RealDatasetLoader()
        transactions = loader.load_and_preprocess()
        
        return jsonify({
            'dataset_loaded': True,
            'transaction_count': len(transactions),
            'sample_transactions': transactions[:3]
        })
    except Exception as e:
        return jsonify({
            'dataset_loaded': False,
            'error': str(e)
        })

@debug_bp.route('/test-security')
@login_required
def test_security():
    security = AdvancedSecurity()
    
    # Test encryption
    test_data = "Sensitive Credit Card Info"
    encrypted = security.encrypt_sensitive_data(test_data)
    decrypted = security.decrypt_sensitive_data(encrypted)
    
    # Test input sanitization
    dangerous_input = "'; DROP TABLE users; --"
    sanitized = security.sanitize_input(dangerous_input)
    
    # Test JWT
    token = security.generate_jwt_token(current_user.id, current_user.username, current_user.role)
    
    return jsonify({
        'encryption_works': decrypted == test_data,
        'sanitization_works': ';' not in sanitized,
        'jwt_works': bool(token),
        'encrypted_sample': encrypted,
        'sanitized_sample': sanitized
    })

@debug_bp.route('/test-mcp')
@login_required
def test_mcp():
    try:
        from protocols.mcp_protocol import MCPProtocol
        mcp = MCPProtocol()
        
        test_message = {
            'action': 'test_communication',
            'user_id': current_user.id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Test sending message
        response = mcp.send_to_architect(test_message, 'tester')
        
        return jsonify({
            'mcp_working': True,
            'response': response,
            'message_sent': test_message
        })
    except Exception as e:
        return jsonify({
            'mcp_working': False,
            'error': str(e)
        })

@debug_bp.route('/test-gemini-detailed')
@login_required
def test_gemini_detailed():
    """Detailed test of Gemini integration"""
    try:
        from agents.gemini_processor import GeminiProcessor
        gemini = GeminiProcessor()
        
        test_cases = [
            {"note": "Monthly salary payment from company", "amount": 5000, "expected": "salary"},
            {"note": "Groceries at Walmart", "amount": 150, "expected": "food"},
            {"note": "Electricity bill payment", "amount": 120, "expected": "bills"},
            {"note": "Gas for car", "amount": 60, "expected": "transport"},
            {"note": "Dinner at restaurant", "amount": 80, "expected": "food"}
        ]
        
        results = {
            'gemini_available': gemini.available,
            'model_used': gemini.model_name,
            'test_results': []
        }
        
        if gemini.available:
            for test_case in test_cases:
                category = gemini.categorize_transaction(
                    test_case["note"], 
                    test_case["amount"]
                )
                results['test_results'].append({
                    'note': test_case["note"],
                    'amount': test_case["amount"],
                    'expected': test_case["expected"],
                    'actual': category,
                    'match': category == test_case["expected"]
                })
            
            # Test insights generation
            test_transactions = [
                {'date': '2024-01-01', 'type': 'income', 'category': 'salary', 'amount': 5000, 'note': 'Salary'},
                {'date': '2024-01-02', 'type': 'expense', 'category': 'food', 'amount': 150, 'note': 'Groceries'},
                {'date': '2024-01-03', 'type': 'expense', 'category': 'bills', 'amount': 120, 'note': 'Electricity'},
                {'date': '2024-01-04', 'type': 'expense', 'category': 'transport', 'amount': 60, 'note': 'Gas'},
                {'date': '2024-01-05', 'type': 'expense', 'category': 'entertainment', 'amount': 80, 'note': 'Movies'}
            ]
            
            insights = gemini.generate_financial_insights(test_transactions)
            results['insights_test'] = {
                'success': bool(insights),
                'preview': insights[:200] + "..." if len(insights) > 200 else insights
            }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            'gemini_available': False,
            'error': str(e)
        })

@debug_bp.route('/test-gaming-categories')
@login_required
def test_gaming_categories():
    """Test how gaming transactions are categorized"""
    from agents.nlp_processor import NLPProcessor
    nlp = NLPProcessor()
    
    gaming_transactions = [
        {"note": "Clash of Clans gems purchase", "amount": 4.99},
        {"note": "PlayStation Store wallet top up", "amount": 20.00},
        {"note": "Steam game sale", "amount": 29.99},
        {"note": "Xbox Game Pass subscription", "amount": 14.99},
        {"note": "Mobile game in-app purchase", "amount": 2.99},
        {"note": "Nintendo Switch online", "amount": 3.99},
        {"note": "Gaming mouse from Best Buy", "amount": 59.99},
        {"note": "Internet bill for online gaming", "amount": 79.99},
    ]
    
    results = []
    for transaction in gaming_transactions:
        ai_category = nlp.categorize_transaction_llm(
            transaction["note"], 
            transaction["amount"]
        )
        basic_category = nlp.categorize_transaction_basic(
            transaction["note"], 
            transaction["amount"]
        )
        
        results.append({
            'note': transaction["note"],
            'amount': transaction["amount"],
            'ai_category': ai_category,
            'basic_category': basic_category,
            'match': ai_category == basic_category
        })
    
    return jsonify({
        'llm_available': nlp.llm_available,
        'gemini_model': nlp.gemini_processor.model_name if nlp.gemini_processor else None,
        'gaming_transactions': results
    })

@debug_bp.route('/add-test-data')
@login_required
def add_test_data():
    """Add sample data to demonstrate all features"""
    test_transactions = [
        # Expenses (for budget tracking, analytics)
        {'date': datetime(2024, 10, 1), 'type': 'expense', 'category': '', 'amount': 150.00, 'note': 'Groceries at Walmart'},
        {'date': datetime(2024, 10, 2), 'type': 'expense', 'category': '', 'amount': 45.00, 'note': 'Dinner at pizza restaurant'},
        {'date': datetime(2024, 10, 3), 'type': 'expense', 'category': '', 'amount': 80.00, 'note': 'Gas for car'},
        {'date': datetime(2024, 10, 5), 'type': 'expense', 'category': '', 'amount': 15.99, 'note': 'Netflix subscription'},
        {'date': datetime(2024, 10, 7), 'type': 'expense', 'category': '', 'amount': 4.99, 'note': 'Clash of Clans gems'},
        
        # Income
        {'date': datetime(2024, 10, 1), 'type': 'income', 'category': '', 'amount': 500.00, 'note': 'Freelance work'},
    ]
    
    added = 0
    for data in test_transactions:
        # Let AI categorize by leaving category empty
        transaction = Transaction(
            user_id=current_user.id,
            t_type=data['type'],
            category='',  # EMPTY - let AI categorize
            amount=data['amount'],
            date=data['date'],
            note=data['note']
        )
        
        # AI categorization will happen here
        from agents.nlp_processor import NLPProcessor
        nlp_processor = NLPProcessor()
        if not transaction.category:
            transaction.category = nlp_processor.categorize_transaction_llm(
                transaction.note, transaction.amount
            )
            print(f"🤖 AI categorized '{transaction.note}' as: {transaction.category}")
        
        db.session.add(transaction)
        added += 1
    
    db.session.commit()
    
    return jsonify({
        'success': True,
        'added': added,
        'message': 'Test data added with AI categorization'
    })

@debug_bp.route('/api/feature-status')
@login_required
def api_feature_status():
    """API endpoint for feature status"""
    status = {}
    
    # Check NLP
    try:
        from agents.nlp_processor import NLPProcessor
        nlp = NLPProcessor()
        status['nlp'] = {
            'working': nlp.nlp is not None,
            'status': 'Working' if nlp.nlp else 'Not Working',
            'details': 'spaCy model loaded' if nlp.nlp else 'spaCy model not found'
        }
        status['llm'] = {
            'working': nlp.llm_available,
            'status': 'Working' if nlp.llm_available else 'Not Configured',
            'details': f'Gemini: {nlp.gemini_processor.model_name if nlp.gemini_processor else "Not available"}'
        }
    except Exception as e:
        status['nlp'] = {'working': False, 'status': 'Error', 'details': str(e)}
        status['llm'] = {'working': False, 'status': 'Error', 'details': str(e)}
    
    # Check Real Dataset
    try:
        from data_loader import RealDatasetLoader
        loader = RealDatasetLoader()
        transactions = loader.load_and_preprocess()
        status['dataset'] = {
            'working': len(transactions) > 0,
            'status': f'{len(transactions)} records',
            'details': f'Loaded {len(transactions)} transactions from dataset'
        }
    except Exception as e:
        status['dataset'] = {'working': False, 'status': 'Error', 'details': str(e)}
    
    # Check Security
    try:
        from core.advanced_security import AdvancedSecurity
        security = AdvancedSecurity()
        # Test encryption
        test_data = "test"
        encrypted = security.encrypt_sensitive_data(test_data)
        decrypted = security.decrypt_sensitive_data(encrypted)
        status['security'] = {
            'working': decrypted == test_data,
            'status': 'Active',
            'details': 'Encryption/decryption working'
        }
    except Exception as e:
        status['security'] = {'working': False, 'status': 'Error', 'details': str(e)}
    
    # Check Agent Communication (MCP)
    try:
        from protocols.mcp_protocol import MCPProtocol
        mcp = MCPProtocol()
        status['mcp'] = {
            'working': True,
            'status': 'Online',
            'details': 'MCP protocol available'
        }
    except Exception as e:
        status['mcp'] = {'working': False, 'status': 'Offline', 'details': str(e)}
    
    # Check Predictive Analytics
    try:
        from agents.insight_generator import InsightGeneratorAgent
        agent = InsightGeneratorAgent()
        status['analytics'] = {
            'working': True,
            'status': 'Available',
            'details': 'Predictive analytics engine ready'
        }
    except Exception as e:
        status['analytics'] = {'working': False, 'status': 'Error', 'details': str(e)}
    
    return jsonify(status)

@debug_bp.route('/test-ai-categorization')
@login_required
def test_ai_categorization():
    """Test AI categorization with various transaction examples"""
    from agents.nlp_processor import NLPProcessor
    nlp = NLPProcessor()
    
    test_cases = [
        {"note": "Clash of Clans gems purchase", "amount": 4.99, "type": "expense"},
        {"note": "Monthly salary from company", "amount": 5000, "type": "income"},
        {"note": "Groceries at supermarket", "amount": 85.50, "type": "expense"},
        {"note": "Electricity bill payment", "amount": 120, "type": "expense"},
        {"note": "Gas for car at Shell station", "amount": 45, "type": "expense"},
        {"note": "Netflix subscription", "amount": 15.99, "type": "expense"},
        {"note": "Dinner at Italian restaurant", "amount": 65, "type": "expense"},
        {"note": "University tuition fees", "amount": 1200, "type": "expense"},
        {"note": "Doctor visit and medicine", "amount": 150, "type": "expense"},
        {"note": "Amazon online shopping", "amount": 89.99, "type": "expense"},
    ]
    
    results = []
    for test in test_cases:
        ai_category = nlp.categorize_transaction_llm(test["note"], test["amount"])
        basic_category = nlp.categorize_transaction_basic(test["note"], test["amount"])
        
        results.append({
            'note': test["note"],
            'amount': test["amount"],
            'type': test["type"],
            'ai_category': ai_category,
            'basic_category': basic_category,
            'ai_working': ai_category != 'other'
        })
    
    return jsonify({
        'llm_available': nlp.llm_available,
        'gemini_model': nlp.gemini_processor.model_name if nlp.gemini_processor else None,
        'test_results': results
    })

@debug_bp.route('/test-data-loader')
@login_required
def test_data_loader():
    """Test the real dataset loader"""
    try:
        from data_loader import RealDatasetLoader
        loader = RealDatasetLoader()
        
        # Load data
        transactions = loader.load_and_preprocess()
        stats = loader.get_dataset_stats()
        
        return jsonify({
            'success': True,
            'dataset_loaded': True,
            'transaction_count': len(transactions),
            'stats': stats,
            'sample_transactions': transactions[:5]  # First 5 transactions
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })