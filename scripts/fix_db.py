"""
fix_db.py
─────────
Run this ONCE to add all new columns and tables to existing database.

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

# ── Add is_admin to users (if missing) ───────────────────────────────────
try:
    cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0 NOT NULL")
    conn.commit()
    print("✅ Column 'is_admin' added to users")
except sqlite3.OperationalError:
    print("✅ Column 'is_admin' already exists")

# ── Create contact_messages table ────────────────────────────────────────
cursor.execute("""
    CREATE TABLE IF NOT EXISTS contact_messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        name       VARCHAR(100) NOT NULL,
        email      VARCHAR(150) NOT NULL,
        subject    VARCHAR(200) NOT NULL DEFAULT 'No subject',
        message    TEXT NOT NULL,
        is_read    BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
print("✅ Table 'contact_messages' ready")

# ── Create newsletter_subscribers table ──────────────────────────────────
cursor.execute("""
    CREATE TABLE IF NOT EXISTS newsletter_subscribers (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        email      VARCHAR(150) UNIQUE NOT NULL,
        is_active  BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
print("✅ Table 'newsletter_subscribers' ready")

# ── Create farmers table ──────────────────────────────────────────────────
cursor.execute("""
    CREATE TABLE IF NOT EXISTS farmers (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        VARCHAR(100) NOT NULL,
        phone       VARCHAR(20) UNIQUE NOT NULL,
        district    VARCHAR(50) DEFAULT 'ilam',
        language    VARCHAR(5)  DEFAULT 'ne',
        is_active   BOOLEAN DEFAULT 1,
        last_sms_at DATETIME,
        created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()
print("✅ Table 'farmers' ready")

# ── Show current tables ───────────────────────────────────────────────────
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print(f"\n📋 All tables: {', '.join(tables)}")

conn.close()
print("\n✅ Database migration complete!")
print("   Run: python make_admin.py  to grant admin access")