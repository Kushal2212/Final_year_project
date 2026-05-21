# Make a user admin by their email address.

import sqlite3
import os

db_path = os.path.join("database", "cardamom.db")
conn    = sqlite3.connect(db_path)
cursor  = conn.cursor()

# Show all users
cursor.execute("SELECT id, name, email, is_admin FROM users")
users = cursor.fetchall()

if not users:
    print("No users found. Register an account first.")
    conn.close()
    exit()

print("\n=== CURRENT USERS ===")
for u in users:
    admin = " (ADMIN)" if u[3] else ""
    print(f"  ID:{u[0]}  {u[1]}  {u[2]}{admin}")

print("\nEnter the email of the user to make admin:")
email = input("Email: ").strip()

cursor.execute("SELECT id, name FROM users WHERE email = ?", (email,))
user = cursor.fetchone()

if not user:
    print(f"User with email '{email}' not found.")
else:
    cursor.execute("UPDATE users SET is_admin = 1 WHERE email = ?", (email,))
    conn.commit()
    print(f"\n✅ {user[1]} is now an admin!")
    print("   Visit http://127.0.0.1:5000/admin to access the dashboard")

conn.close()