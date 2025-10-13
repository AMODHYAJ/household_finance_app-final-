import re
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, Response
from flask_login import login_required, current_user
import pytesseract
from app import db
from app.models.forms import TransactionForm
from core.database import Transaction
from agents.data_collector import DataCollectorAgent
import pandas as pd
import os
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import csv

transactions_bp = Blueprint('transactions', __name__)
collector = DataCollectorAgent(use_llm=True)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- DASHBOARD -------------------
@transactions_bp.route('/')
@transactions_bp.route('/dashboard')
@login_required
def dashboard():
    transactions = collector.get_user_transactions(current_user.id)
    
    if not transactions:
        return render_template('dashboard.html', title='Dashboard',
                             transactions=[], metrics={}, anomalies=[], summary={},
                             show_welcome=True)

    df = collector.get_transactions_df(current_user.id)
    
    # Calculate metrics
    total_income = df[df['type'].str.lower() == 'income']['amount'].sum()
    total_expense = df[df['type'].str.lower() == 'expense']['amount'].sum()
    savings = total_income - total_expense
    savings_rate = (savings / total_income * 100) if total_income > 0 else 0

    metrics = {
        'total_income': total_income,
        'total_expense': total_expense,
        'savings': savings,
        'savings_rate': savings_rate,
        'transaction_count': len(transactions)
    }

    # Get recent transactions (latest 10)
    recent_transactions = sorted(transactions, key=lambda x: x.date, reverse=True)[:10]
    
    # Get anomalies and summary
    anomalies = collector.detect_anomalies(current_user.id)
    summary = collector.generate_summary(current_user.id, period='week')

    return render_template('dashboard.html',
                         title='Dashboard',
                         metrics=metrics,
                         transactions=recent_transactions,
                         anomalies=anomalies,
                         summary=summary,
                         show_welcome=False)

# ------------------- ADD TRANSACTION -------------------
@transactions_bp.route('/add-transaction', methods=['GET', 'POST'])
@login_required
def add_transaction():
    form = TransactionForm()

    if form.validate_on_submit():
        note = form.note.data
        amount = float(form.amount.data)
        
        # Auto-categorize if category is empty
        category = form.category.data
        if not category or category.strip() == '':
            category = collector.suggest_category(note)
        
        # Get transaction type
        t_type = form.t_type.data
        
        # Handle date
        date_input = form.date.data
        if not date_input:
            date_input = datetime.today().date()

        # Validate transaction
        try:
            collector.validate_transaction(amount, t_type)
        except ValueError as e:
            flash(str(e), 'danger')
            return render_template('add_transaction.html', title='Add Transaction', form=form, collector=collector)

        # Save transaction
        transaction = Transaction(
            user_id=current_user.id,
            t_type=t_type,
            category=category,
            amount=amount,
            date=date_input,
            note=note or None
        )
        
        db.session.add(transaction)
        db.session.commit()

        flash(f'Transaction added successfully! Categorized as "{category}"', 'success')
        return redirect(url_for('transactions.view_transactions'))

    return render_template('add_transaction.html', title='Add Transaction', form=form, collector=collector)

# ------------------- VIEW TRANSACTIONS -------------------
@transactions_bp.route('/view-transactions')
@login_required
def view_transactions():
    # Get page number from request args
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of transactions per page
    
    # Get paginated transactions for user
    transactions_pagination = Transaction.query.filter_by(user_id=current_user.id)\
        .order_by(Transaction.date.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    transactions = transactions_pagination.items

    if not transactions:
        return render_template('view_transactions.html',
                             transactions=transactions_pagination,  # Pass pagination object
                             anomalies=[],
                             anomaly_ids=[],
                             llm_explanations={},
                             summary={})

    # Detect anomalies
    anomalies = collector.detect_anomalies(current_user.id)
    anomaly_ids = [a['id'] for a in anomalies]

    # Get AI explanations for anomalies
    llm_explanations = {}
    for txn_dict in anomalies:
        txn = Transaction.query.get(txn_dict['id'])
        if txn:
            llm_explanations[txn.id] = collector.explain_anomaly(txn)

    # Generate summary
    summary = collector.generate_summary(current_user.id, period='week')

    return render_template(
        'view_transactions.html',
        transactions=transactions_pagination,  # Pass the pagination object, not just items
        anomalies=anomalies,
        anomaly_ids=anomaly_ids,
        llm_explanations=llm_explanations,
        summary=summary
    )

# ------------------- OCR UPLOAD -------------------
@transactions_bp.route('/api/ocr-upload', methods=['POST'])
@login_required
def ocr_upload():
    if 'receipt' not in request.files or request.files['receipt'].filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['receipt']
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    print(f"📁 Processing uploaded file: {filename}")

    try:
        # Import OCR utilities
        try:
            from app.models.ocr_utils import extract_text_from_image, parse_receipt_text, check_tesseract_available
        except ImportError as e:
            print(f"❌ OCR utils import error: {e}")
            # Clean up file
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'OCR functionality not available'}), 500
        
        # Check if Tesseract is available
        if not check_tesseract_available():
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Tesseract OCR is not installed or not accessible at the configured path'}), 400
        
        # Check file type
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.pdf']
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'File type {file_ext} not supported. Use: {", ".join(allowed_extensions)}'}), 400
        
        print(f"🔍 Running OCR on file: {filename}")
        
        # Run OCR
        text = extract_text_from_image(file_path)
        if not text or text.strip() == "":
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'No text could be extracted from the image/PDF. Try a clearer image or different file.'}), 400
            
        print(f"📄 OCR extracted {len(text)} characters")
        
        parsed = parse_receipt_text(text)

        # If no amount was found, consider it unsuccessful
        if not parsed.get('success', False) or parsed.get('amount', 0) <= 0:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'error': 'Could not extract amount from receipt. Please enter amount manually.',
                'extracted_text': text[:500] + '...' if len(text) > 500 else text
            }), 400

        # Clean amount
        try:
            parsed['amount'] = float(parsed['amount'])
            if parsed['amount'] <= 0:
                parsed['amount'] = 0.0
        except (ValueError, TypeError):
            parsed['amount'] = 0.0

        # Clean date - use today if no date found
        parsed_date = None
        if parsed.get('date'):
            try:
                parsed_date = datetime.strptime(parsed['date'], '%Y-%m-%d').date()
            except:
                parsed_date = datetime.today().date()
        else:
            parsed_date = datetime.today().date()
        parsed['date'] = parsed_date.strftime('%Y-%m-%d')

        # Use AI categorization if available, otherwise use rule-based from OCR
        if parsed.get('note') and parsed['note'] != "Receipt scan":
            try:
                ai_category = collector.suggest_category(parsed['note'])
                if ai_category and ai_category != "Other":
                    parsed['category'] = ai_category
                    print(f"🤖 AI categorized as: {ai_category}")
            except Exception as e:
                print(f"AI categorization failed: {e}")

        # Use AI transaction type suggestion
        if parsed.get('note') and parsed['note'] != "Receipt scan":
            try:
                ai_type = collector.suggest_transaction_type(parsed['note'])
                parsed['t_type'] = ai_type
                print(f"🤖 AI transaction type: {ai_type}")
            except Exception as e:
                print(f"AI transaction type failed: {e}")

        # Clean up file
        if os.path.exists(file_path):
            os.remove(file_path)

        print(f"✅ OCR processing successful: {parsed}")

        return jsonify({
            'amount': parsed['amount'],
            'date': parsed['date'],
            'note': parsed.get('note', ''),
            'category': parsed['category'],
            't_type': parsed['t_type'],
            'success': True,
            'message': f'Successfully extracted: ${parsed["amount"]:.2f} for {parsed["category"]}'
        })

    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        print(f"❌ OCR processing error: {e}")
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500
    
    # ------------------- SEARCH TRANSACTIONS -------------------


@transactions_bp.route('/api/search-transactions')
@login_required
def search_transactions():
    query = request.args.get('q', '').strip().lower()
    transactions = collector.get_user_transactions(current_user.id)
    anomalies = collector.detect_anomalies(current_user.id)
    anomaly_ids = [a['id'] for a in anomalies]

    # Prepare AI explanations
    llm_explanations = {}
    for a in anomalies:
        txn = Transaction.query.get(a['id'])
        if txn:
            llm_explanations[txn.id] = collector.explain_anomaly(txn)

    # If query is empty, return all transactions
    if not query:
        # Remove duplicates before returning
        unique_transactions = remove_duplicate_transactions(transactions)
        results = [{
            "id": t.id,
            "date": t.date.strftime('%Y-%m-%d'),
            "type": t.t_type,
            "category": t.category,
            "amount": t.amount,
            "note": t.note or '-',
            "anomaly": t.id in anomaly_ids,
            "ai_insight": llm_explanations.get(t.id, '-')
        } for t in unique_transactions]
        return jsonify({"results": results})

    # Enhanced natural language query parsing
    parsed_query = parse_natural_language_query(query)
    
    print(f"🔍 Searching for: '{query}'")
    print(f"📝 Parsed query: {parsed_query}")
    
    # Filter transactions based on parsed query
    matches = []
    seen_transactions = set()
    
    for t in transactions:
        # Skip duplicates
        transaction_key = f"{t.date}_{t.amount}_{t.note}"
        if transaction_key in seen_transactions:
            continue
        seen_transactions.add(transaction_key)
        
        score = calculate_relevance_score(t, parsed_query, query)
        
        if score > 20:  # Increased threshold to filter out weak matches
            matches.append({
                "id": t.id,
                "date": t.date.strftime('%Y-%m-%d'),
                "type": t.t_type,
                "category": t.category,
                "amount": t.amount,
                "note": t.note or '-',
                "anomaly": t.id in anomaly_ids,
                "ai_insight": llm_explanations.get(t.id, '-'),
                "score": score
            })

    # Sort by relevance score (highest first)
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"✅ Found {len(matches)} relevant transactions")
    
    return jsonify({"results": matches})

def remove_duplicate_transactions(transactions):
    """Remove duplicate transactions based on date, amount, and note"""
    seen = set()
    unique_transactions = []
    
    for t in transactions:
        key = f"{t.date}_{t.amount}_{t.note}"
        if key not in seen:
            seen.add(key)
            unique_transactions.append(t)
    
    return unique_transactions

def parse_natural_language_query(query):
    """Parse natural language queries into structured filters"""
    parsed = {
        "food_items": [],
        "time_period": None,
        "categories": [],
        "amount_filters": [],
        "exact_phrases": [],
        "is_food_query": False
    }
    
    # Food-related keywords
    food_keywords = ['pizza', 'burger', 'coffee', 'restaurant', 'dining', 'meal', 
                    'food', 'groceries', 'lunch', 'dinner', 'breakfast', 'ate', 'eat',
                    'cafe', 'restaurant', 'food', 'eating']
    
    # Check if this is a food-related query
    if any(food in query for food in food_keywords):
        parsed["is_food_query"] = True
    
    # Extract food items
    for food in food_keywords:
        if food in query:
            parsed["food_items"].append(food)
    
    # Time-related keywords
    time_keywords = {
        'today': 0,
        'yesterday': 1,
        'this week': 7,
        'last week': 14,
        'this month': 30,
        'last month': 60
    }
    
    # Extract time periods
    for time_key, days in time_keywords.items():
        if time_key in query:
            parsed["time_period"] = time_key
            break
    
    # Category mapping
    category_keywords = {
        'food': ['food', 'eating', 'restaurant', 'cafe', 'coffee', 'meal', 'dining'],
        'transport': ['transport', 'uber', 'taxi', 'bus', 'train', 'gas'],
        'shopping': ['shopping', 'amazon', 'walmart', 'purchase'],
        'entertainment': ['entertainment', 'movie', 'netflix', 'game']
    }
    
    # Extract categories
    for category, keywords in category_keywords.items():
        if any(keyword in query for keyword in keywords):
            parsed["categories"].append(category)
    
    # Extract amount filters
    amount_patterns = [
        (r'over\s+\$?(\d+)', 'gt'),
        (r'more than\s+\$?(\d+)', 'gt'),
        (r'under\s+\$?(\d+)', 'lt'),
        (r'less than\s+\$?(\d+)', 'lt')
    ]
    
    for pattern, op in amount_patterns:
        match = re.search(pattern, query)
        if match:
            parsed["amount_filters"].append({
                "operator": op,
                "value": float(match.group(1))
            })
    
    # Extract exact phrases (for questions like "when did i")
    question_patterns = [
        r'when did i (.*)',
        r'where did i (.*)', 
        r'how much did i (.*)',
        r'show me (.*)',
        r'find (.*)'
    ]
    
    for pattern in question_patterns:
        match = re.search(pattern, query)
        if match:
            action_phrase = match.group(1)
            parsed["exact_phrases"].append(action_phrase)
            # Also extract food items from the action phrase
            for food in food_keywords:
                if food in action_phrase:
                    parsed["food_items"].append(food)
    
    return parsed

def calculate_relevance_score(transaction, parsed_query, original_query):
    """Calculate how relevant a transaction is to the search query"""
    score = 0
    note = (transaction.note or '').lower()
    category = transaction.category.lower()
    
    # Exact phrase matching (for questions like "when did i eat pizza")
    for phrase in parsed_query["exact_phrases"]:
        if phrase in note:
            score += 200  # Very high score for exact phrase match
            print(f"🎯 Exact phrase match: '{phrase}' in '{note}'")
    
    # For "when did I eat pizza" specifically
    if "pizza" in original_query and "pizza" in note:
        score += 150
        print(f"🍕 Pizza match: '{note}'")
    
    # Food item matching - only if query is food-related
    if parsed_query["food_items"]:
        for food_item in parsed_query["food_items"]:
            if food_item in note:
                score += 80  # Increased score for direct food match
            if food_item in category:
                score += 60
    
    # Category matching - only boost if query mentions categories
    if parsed_query["categories"]:
        if category in parsed_query["categories"]:
            score += 40
    
    # Penalize unrelated categories for food queries
    if parsed_query["food_items"] and category not in ['food', 'dining']:
        score -= 50  # Penalize non-food categories for food queries
    
    # Time period filtering
    if parsed_query["time_period"]:
        transaction_date = transaction.date
        today = datetime.today().date()
        
        if parsed_query["time_period"] == 'today' and transaction_date == today:
            score += 30
        elif parsed_query["time_period"] == 'yesterday' and transaction_date == today - timedelta(days=1):
            score += 30
        elif parsed_query["time_period"] == 'this week':
            week_start = today - timedelta(days=today.weekday())
            if transaction_date >= week_start:
                score += 20
    
    # Basic keyword matching (lower priority)
    query_words = original_query.split()
    for word in query_words:
        if len(word) > 3:  # Only match words longer than 3 characters
            if word in note:
                score += 3
            if word in category:
                score += 2
    
    # Debug output
    if score > 50:
        print(f"📊 Transaction '{note}' scored: {score}")
    
    return score
# ------------------- NLP SMART TRANSACTION -------------------
@transactions_bp.route('/api/nlp-transaction', methods=['POST'])
@login_required
def nlp_transaction():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "Please enter a description."}), 400

        lower = text.lower()

        # Detect transaction type
        if any(word in lower for word in ["spent", "bought", "paid", "purchase", "pay"]):
            t_type = "Expense"
        elif any(word in lower for word in ["received", "salary", "earned", "income", "got", "credited"]):
            t_type = "Income"
        else:
            t_type = "Expense"  # Default

        # Extract amount
        import re
        amount_match = re.search(r"(\d+(?:\.\d{1,2})?)", text)
        amount = float(amount_match.group(1)) if amount_match else 0.0

        if amount <= 0:
            return jsonify({"error": "Could not detect valid amount"}), 400

        # Extract category using AI
        category = collector.suggest_category(text)

        # Extract date references
        today = datetime.now().date()
        if "yesterday" in lower:
            date = today - timedelta(days=1)
        elif "last week" in lower:
            date = today - timedelta(days=7)
        else:
            date = today

        # Create note (first 100 chars)
        note = text[:100]

        # Save transaction
        transaction = Transaction(
            user_id=current_user.id,
            t_type=t_type,
            category=category,
            amount=amount,
            date=date,
            note=note,
        )
        
        db.session.add(transaction)
        db.session.commit()

        return jsonify({
            "message": f"✅ {t_type} of ${amount:.2f} added under {category}.",
            "transaction": {
                "type": t_type,
                "amount": amount,
                "category": category,
                "note": note,
                "date": date.strftime("%Y-%m-%d")
            }
        }), 201

    except Exception as e:
        print(f"NLP transaction error: {e}")
        return jsonify({"error": "Failed to process transaction"}), 500

# ------------------- EXPORT CSV -------------------
@transactions_bp.route('/export-csv')
@login_required
def export_csv():
    transactions = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc()).all()

    def generate():
        header = ['Date', 'Type', 'Category', 'Amount', 'Note']
        yield ','.join(header) + '\n'

        for t in transactions:
            row = [
                t.date.strftime('%Y-%m-%d'),
                t.t_type,
                t.category,
                f"{t.amount:.2f}",
                f'"{t.note or ""}"'  # Quote note to handle commas
            ]
            yield ','.join(row) + '\n'

    return Response(
        generate(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=transactions.csv'}
    )

@transactions_bp.route('/api/ai-status')
@login_required
def ai_status():
    """Check AI processor status"""
    status = collector.get_ai_status()
    return jsonify(status)

# ------------------- AI SEARCH DEMO -------------------
@transactions_bp.route('/ai-search-demo')
@login_required
def ai_search_demo():
    """Demo page for AI-powered search"""
    return render_template('ai_search.html', title='AI Transaction Search')

@transactions_bp.route('/debug-duplicates')
@login_required
def debug_duplicates():
    """Check for duplicate transactions"""
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    
    # Find duplicates
    seen = set()
    duplicates = []
    unique_count = 0
    
    for t in transactions:
        transaction_key = f"{t.date}_{t.amount}_{t.note}"
        if transaction_key in seen:
            duplicates.append({
                'id': t.id,
                'date': str(t.date),
                'amount': t.amount,
                'note': t.note,
                'category': t.category
            })
        else:
            seen.add(transaction_key)
            unique_count += 1
    
    return jsonify({
        'total_transactions': len(transactions),
        'unique_transactions': unique_count,
        'duplicates_found': len(duplicates),
        'duplicates': duplicates
    })