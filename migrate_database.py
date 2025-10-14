from app import create_app, db
from core.database import Transaction

app = create_app()

with app.app_context():
    print("🔧 Starting database migration...")
    
    # Get all transactions
    transactions = Transaction.query.all()
    print(f"📊 Found {len(transactions)} transactions to migrate")
    
    for transaction in transactions:
        # Copy t_type to type
        if hasattr(transaction, 't_type') and not hasattr(transaction, 'type'):
            transaction.type = transaction.t_type
            print(f"🔄 Migrated transaction {transaction.id}: {transaction.t_type} → {transaction.type}")
    
    try:
        db.session.commit()
        print("✅ Database migration completed successfully!")
    except Exception as e:
        db.session.rollback()
        print(f"❌ Migration failed: {e}")