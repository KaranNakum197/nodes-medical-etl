"""
Schema migrations entry point.

Run with:
    python -m database.schema
"""

from .connection import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created / verified.")
