"""
history_db.py  –  Nutrifexa-AI
Manages the SQLite database for persisting food prediction history.
"""

import os
import sqlite3
import csv
import datetime
from io import StringIO

DB_PATH = "history.db"

def init_db():
    """Initializes the database and creates the tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            food_name TEXT,
            calories REAL,
            confidence REAL,
            date TEXT,
            time TEXT,
            protein REAL,
            carbs REAL,
            fat REAL,
            weight REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            gender TEXT,
            height REAL,
            activity_level TEXT,
            target_weight REAL,
            initial_weight REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            weight REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS community_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            image_path TEXT,
            food_name TEXT,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            health_rating TEXT,
            recipe_title TEXT,
            recipe_instructions TEXT,
            likes_count INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS water_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            glasses INTEGER DEFAULT 0
        )
    """)
    
    # Check if community_posts table is empty, if so, seed it
    cursor.execute("SELECT COUNT(*) FROM community_posts")
    if cursor.fetchone()[0] == 0:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        # Seed 1: Sambar
        cursor.execute("""
            INSERT INTO community_posts (username, image_path, food_name, calories, protein, carbs, fat, health_rating, recipe_title, recipe_instructions, likes_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "HealthyEats",
            "/static/images/demo_sambar.jpg",
            "Sambar",
            120.0, 4.5, 18.0, 3.2, "A+",
            "Grandma's Low-Fat Lentil Sambar",
            "1. Boil yellow lentils (toor dal) with turmeric till soft.\n2. Add chopped vegetables (drumstick, pumpkin, tomato, shallots) and cook till tender.\n3. Pour in tamarind extract and add 2 tbsp of sambar powder.\n4. Simmer for 15 minutes.\n5. Temper with mustard seeds, curry leaves, and a pinch of hing (asafoetida) using minimal ghee/oil.",
            24,
            now_str
        ))
        
        # Seed 2: Chicken Curry
        cursor.execute("""
            INSERT INTO community_posts (username, image_path, food_name, calories, protein, carbs, fat, health_rating, recipe_title, recipe_instructions, likes_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "GymFueler",
            "/static/images/demo_chicken.jpg",
            "Chicken Curry",
            210.0, 28.0, 5.5, 8.0, "A",
            "High-Protein Lean Chicken Breast Curry",
            "1. Marinate chicken breast cubes in low-fat yogurt, ginger-garlic paste, and spices (cumin, coriander, garam masala) for 30 minutes.\n2. Sauté chopped onions and tomatoes in a pan with 1 tsp of olive oil.\n3. Add marinated chicken and sauté until the chicken is sealed and opaque.\n4. Pour in half a cup of water, cover and simmer for 12 minutes.\n5. Garnish with fresh coriander leaves.",
            42,
            now_str
        ))
    conn.commit()
    conn.close()

def add_prediction(image_path, food_name, calories, confidence, date, time, protein, carbs, fat, weight):
    """Adds a new prediction entry to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (image_path, food_name, calories, confidence, date, time, protein, carbs, fat, weight)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (image_path, food_name, calories, confidence, date, time, protein, carbs, fat, weight))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def get_predictions(search_query=None):
    """Retrieves all predictions, optionally filtered by a search query (food_name)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if search_query:
        cursor.execute("""
            SELECT id, image_path, food_name, calories, confidence, date, time, protein, carbs, fat, weight 
            FROM history 
            WHERE food_name LIKE ? 
            ORDER BY id DESC
        """, (f"%{search_query}%",))
    else:
        cursor.execute("""
            SELECT id, image_path, food_name, calories, confidence, date, time, protein, carbs, fat, weight 
            FROM history 
            ORDER BY id DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    
    # Map to dictionary list
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "image_path": r[1],
            "food_name": r[2],
            "calories": r[3],
            "confidence": r[4],
            "date": r[5],
            "time": r[6],
            "protein": r[7],
            "carbs": r[8],
            "fat": r[9],
            "weight": r[10]
        })
    return results

def get_prediction_by_id(entry_id):
    """Returns a single prediction entry by its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, food_name, calories, confidence, date, time, protein, carbs, fat, weight FROM history WHERE id = ?", (entry_id,))
    r = cursor.fetchone()
    conn.close()
    if r:
        return {
            "id": r[0],
            "image_path": r[1],
            "food_name": r[2],
            "calories": r[3],
            "confidence": r[4],
            "date": r[5],
            "time": r[6],
            "protein": r[7],
            "carbs": r[8],
            "fat": r[9],
            "weight": r[10]
        }
    return None

def delete_prediction(entry_id):
    """Deletes a prediction entry from the database by its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get the image path to delete the file as well
    cursor.execute("SELECT image_path FROM history WHERE id = ?", (entry_id,))
    row = cursor.fetchone()
    if row:
        img_path = row[0]
        # Avoid deleting default/demo images or system files
        if img_path and os.path.exists(img_path) and "uploads" in img_path:
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Error removing file {img_path}: {e}")
                
    cursor.execute("DELETE FROM history WHERE id = ?", (entry_id,))
    conn.commit()
    conn.close()

def clear_history():
    """Clears all prediction history records and deletes uploaded images."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Select all paths
    cursor.execute("SELECT image_path FROM history")
    rows = cursor.fetchall()
    for row in rows:
        img_path = row[0]
        if img_path and os.path.exists(img_path) and "uploads" in img_path:
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Error removing file {img_path}: {e}")
                
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()

def export_to_csv():
    """Exports prediction history database records into a CSV string."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, food_name, calories, confidence, date, time, protein, carbs, fat, weight FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    
    si = StringIO()
    cw = csv.writer(si)
    # Write header
    cw.writerow(["ID", "Food Name", "Calories (kcal)", "Confidence (%)", "Date", "Time", "Protein (g)", "Carbohydrates (g)", "Fat (g)", "Weight (g)"])
    # Write rows
    for r in rows:
        cw.writerow(r)
    return si.getvalue()

def get_todays_totals(date_str):
    """Retrieves the sum of macros consumed on a specific date."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat), SUM(weight)
        FROM history
        WHERE date = ?
    """, (date_str,))
    row = cursor.fetchone()
    conn.close()
    
    return {
        "calories": round(row[0] or 0.0, 1),
        "protein": round(row[1] or 0.0, 1),
        "carbs": round(row[2] or 0.0, 1),
        "fat": round(row[3] or 0.0, 1),
        "weight": round(row[4] or 0.0, 1)
    }

def update_prediction_name(entry_id, new_name, calories, protein, carbs, fat):
    """Updates a prediction entry's name and macros after a second-opinion correction."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE history
        SET food_name = ?, calories = ?, protein = ?, carbs = ?, fat = ?
        WHERE id = ?
    """, (new_name, calories, protein, carbs, fat, entry_id))
    conn.commit()
    conn.close()

def get_user_profile():
    """Retrieves the user profile, or returns a default dict if none exists."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT age, gender, height, activity_level, target_weight, initial_weight FROM user_profile LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "age": row[0],
            "gender": row[1],
            "height": row[2],
            "activity_level": row[3],
            "target_weight": row[4],
            "initial_weight": row[5]
        }
    return None

def update_user_profile(age, gender, height, activity_level, target_weight, initial_weight):
    """Upserts the user profile (keeps only one row)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM user_profile")
    cursor.execute("""
        INSERT INTO user_profile (age, gender, height, activity_level, target_weight, initial_weight)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (age, gender, height, activity_level, target_weight, initial_weight))
    conn.commit()
    conn.close()

def get_weight_history():
    """Retrieves all weight logs sorted by date ASC."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, date, weight FROM weight_history ORDER BY date ASC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "date": r[1], "weight": r[2]} for r in rows]

def add_weight_log(date_str, weight):
    """Adds or updates a weight log for a specific date using INSERT OR REPLACE."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO weight_history (date, weight) VALUES (?, ?)", (date_str, weight))
    conn.commit()
    conn.close()

def delete_weight_log(log_id):
    """Deletes a weight log by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM weight_history WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()

def get_community_posts():
    """Retrieves all community posts sorted by id DESC."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, username, image_path, food_name, calories, protein, carbs, fat, health_rating, recipe_title, recipe_instructions, likes_count, created_at 
        FROM community_posts 
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    posts = []
    for r in rows:
        posts.append({
            "id": r[0],
            "username": r[1],
            "image_path": r[2],
            "food_name": r[3],
            "calories": r[4],
            "protein": r[5],
            "carbs": r[6],
            "fat": r[7],
            "health_rating": r[8],
            "recipe_title": r[9],
            "recipe_instructions": r[10],
            "likes_count": r[11],
            "created_at": r[12]
        })
    return posts

def add_community_post(username, food_name, calories, protein, carbs, fat, health_rating, image_path, recipe_title, recipe_instructions):
    """Inserts a new community post."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute("""
        INSERT INTO community_posts (username, food_name, calories, protein, carbs, fat, health_rating, image_path, recipe_title, recipe_instructions, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (username, food_name, calories, protein, carbs, fat, health_rating, image_path, recipe_title, recipe_instructions, now_str))
    conn.commit()
    conn.close()

def like_community_post(post_id):
    """Increments the likes count of a post by 1."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE community_posts SET likes_count = likes_count + 1 WHERE id = ?", (post_id,))
    conn.commit()
    conn.close()

def get_todays_water(date_str):
    """Returns total water glasses consumed on a given date (default goal: 8 glasses / 2000ml)."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT glasses FROM water_logs WHERE date = ?", (date_str,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else 0

def log_water(date_str, delta):
    """Adds or subtracts water glasses for a given date."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    current = get_todays_water(date_str)
    new_val = max(0, current + delta)
    cursor.execute("""
        INSERT INTO water_logs (date, glasses) VALUES (?, ?)
        ON CONFLICT(date) DO UPDATE SET glasses = ?
    """, (date_str, new_val, new_val))
    conn.commit()
    conn.close()
    return new_val

