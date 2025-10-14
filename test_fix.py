# test_fix.py
from app import create_app, db
from core.database import Transaction, User
from datetime import datetime

app = create_app()

with app.app_context():
    try:
        # Create a test user first if needed
        test_user = User.query.first()
        if not test_user:
            print("⚠️ No users found. You'll need to register first.")
        else:
            print(f"✅ Found user: {test_user.username}")
            
            # Create a test transaction
            test_txn = Transaction(
                user_id=test_user.id,
                type='expense',  # Using the new 'type' field
                category='food',
                amount=25.50,
                date=datetime.now().date(),
                note='Test transaction after migration'
            )
            db.session.add(test_txn)
            db.session.commit()
            print("✅ Successfully created transaction with 'type' field")
            
            # Query it back
            transactions = Transaction.query.filter_by(user_id=test_user.id).all()
            print(f"✅ Successfully queried {len(transactions)} transactions")
            for txn in transactions:
                print(f"   - ID: {txn.id}, Type: {txn.type}, Category: {txn.category}")
                
    except Exception as e:
        print(f"❌ Error: {e}")