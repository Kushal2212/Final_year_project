"""
fix_db.py
─────────
Run this ONCE to add the is_admin column to your existing database.

Usage:
    python fix_db.py
"""

import sqlite3
import os

db_path = os.path.join("database", "cardamom.db")

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    print("Start the app first: python main.py web")
    exit(1)

conn   = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0 NOT NULL")
    conn.commit()
    print("✅ Column 'is_admin' added to users table")
except sqlite3.OperationalError as e:
    if "duplicate column" in str(e).lower():
        print("✅ Column 'is_admin' already exists — no changes needed")
    else:
        print(f"❌ Error: {e}")

conn.close()
print("\nDone! Now run: python make_admin.py")