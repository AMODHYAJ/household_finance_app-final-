from app import create_app, db
import os

app = create_app()

def safe_reset():
    print("🔄 Safe database reset...")
    
    with app.app_context():
        try:
            # Drop all tables using SQLAlchemy
            print("🗑️  Dropping all tables...")
            db.drop_all()
            print("✅ Tables dropped")
            
            # Create all tables with new schema
            print("📊 Creating new tables...")
            db.create_all()
            print("✅ Tables created with new schema")
            
            # Verify the new schema
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            columns = inspector.get_columns('transactions')
            
            print("📋 Transactions table columns:")
            column_names = []
            for column in columns:
                column_names.append(column['name'])
                print(f"   - {column['name']}: {column['type']}")
            
            # Check if migration was successful
            if 'type' in column_names and 't_type' not in column_names:
                print("🎉 SUCCESS: Database migrated to use 'type' column!")
            elif 't_type' in column_names:
                print("❌ WARNING: Still using 't_type' column - check your models")
            else:
                print("❌ ERROR: Unexpected schema state")
                
        except Exception as e:
            print(f"❌ Reset failed: {e}")
            print("💡 Make sure no other process is using the database")

if __name__ == '__main__':
    safe_reset()