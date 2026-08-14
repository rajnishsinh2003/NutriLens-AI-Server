import os
import json
import time
import uuid
import datetime
import sqlite3
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from ultralytics import YOLO

# Import local modules
import history_db
from model_utils import load_model, predict as model_predict, IMAGE_SIZE, DEVICE

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'nutrilens_secret_key_session'
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=365)

@app.before_request
def make_session_permanent():
    session.permanent = True

# Configuration Paths
WEIGHTS_PATH = 'model/efficientnet_b3_nutrilens.pth'
CLASSES_PATH = 'model/class_names.json'
CLASS_MAP_PATH = 'model/class_indices.json'
NUTRITION_DB_PATH = 'nutrition_db.json'
SUBSTITUTIONS_DB_PATH = 'substitutions_db.json'
BARCODE_DB_PATH = 'barcode_db.json'
YOLO_MODEL_PATH = 'yolov8n.pt'
UPLOAD_FOLDER = 'static/uploads'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models
pytorch_model = None
yolo_model = None
idx_to_class = {}
class_names = []
nutrition_db = {}
substitutions_db = {}
barcode_db = {}

# Relevant COCO classes for food detection (excluding generic container 'bowl')
FOOD_COCO_CLASSES = {
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake'
}

# Anchor sizes for portion weight estimation
FOOD_ANCHORS = {
    "dal":          {"weight": 150.0, "area": 113.1, "type": "liquid", "category": "Curry"},
    "curry":        {"weight": 150.0, "area": 113.1, "type": "liquid", "category": "Curry"},
    "sambar":       {"weight": 150.0, "area": 113.1, "type": "liquid", "category": "Curry"},
    "paneer":       {"weight": 150.0, "area": 113.1, "type": "liquid", "category": "Curry"},
    "chicken":      {"weight": 150.0, "area": 113.1, "type": "liquid", "category": "Curry"},
    "roti":         {"weight": 40.0,  "area": 176.7, "type": "flatbread", "category": "Bread"},
    "chapati":      {"weight": 40.0,  "area": 176.7, "type": "flatbread", "category": "Bread"},
    "naan":         {"weight": 60.0,  "area": 200.0, "type": "flatbread", "category": "Bread"},
    "paratha":      {"weight": 80.0,  "area": 176.7, "type": "flatbread", "category": "Bread"},
    "dosa":         {"weight": 120.0, "area": 314.2, "type": "flatbread", "category": "Bread"},
    "puri":         {"weight": 40.0,  "area": 78.5,  "type": "flatbread", "category": "Bread"},
    "papad":        {"weight": 15.0,  "area": 176.7, "type": "flatbread", "category": "Bread"},
    "biryani":      {"weight": 180.0, "area": 113.1, "type": "solid", "category": "Main Course"},
    "rice":         {"weight": 180.0, "area": 113.1, "type": "solid", "category": "Main Course"},
    "chawal":       {"weight": 180.0, "area": 113.1, "type": "solid", "category": "Main Course"},
    "samosa":       {"weight": 100.0, "area": 36.0,  "type": "solid", "category": "Snacks"},
    "vada":         {"weight": 100.0, "area": 50.2,  "type": "solid", "category": "Snacks"},
    "kebab":        {"weight": 120.0, "area": 80.0,  "type": "solid", "category": "Snacks"},
    "gulab jamun":  {"weight": 50.0,  "area": 19.6,  "type": "solid", "category": "Dessert"},
    "rasmalai":     {"weight": 50.0,  "area": 19.6,  "type": "solid", "category": "Dessert"},
    "jalebi":       {"weight": 60.0,  "area": 35.0,  "type": "solid", "category": "Dessert"},
    "fallback":     {"weight": 100.0, "area": 100.0, "type": "solid", "category": "Snacks"}
}

def load_resources():
    """Load deep learning models and metadata files."""
    global pytorch_model, yolo_model, idx_to_class, class_names, nutrition_db, substitutions_db, barcode_db
    
    # 1. Initialize SQLite Database
    history_db.init_db()
    
    # 2. Load PyTorch model
    model_file = WEIGHTS_PATH
    if not os.path.exists(model_file) and os.path.exists('model/best_model.pth'):
        model_file = 'model/best_model.pth'
        
    class_map_file = CLASS_MAP_PATH
    if os.path.exists(model_file) and os.path.exists(class_map_file):
        try:
            print(f"Loading PyTorch Classifier model from {model_file}...")
            pytorch_model, idx_to_class = load_model(model_file, class_map_file)
            print("PyTorch model loaded successfully!")
        except Exception as e:
            print(f"Error loading PyTorch model: {e}")
    else:
        print(f"PyTorch model not found! Checked: {model_file}")

    # Load class names list
    if os.path.exists(CLASSES_PATH):
        try:
            with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
            print(f"Loaded {len(class_names)} food class names.")
        except Exception as e:
            print(f"Error loading class names list: {e}")
            
    # 3. Load YOLOv8 model
    try:
        print("Loading YOLOv8 object detection model...")
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLOv8 loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLOv8: {e}")
        
    # 4. Load nutrition database
    if os.path.exists(NUTRITION_DB_PATH):
        try:
            with open(NUTRITION_DB_PATH, 'r', encoding='utf-8') as f:
                nutrition_db = json.load(f)
            print(f"Loaded nutrition database with {len(nutrition_db)} items.")
        except Exception as e:
            print(f"Error loading nutrition database: {e}")
    
    # 5. Load substitutions database for Plate Mutator
    if os.path.exists(SUBSTITUTIONS_DB_PATH):
        try:
            with open(SUBSTITUTIONS_DB_PATH, 'r', encoding='utf-8') as f:
                substitutions_db = json.load(f)
            print(f"Loaded substitutions database for AI Plate Mutator.")
        except Exception as e:
            print(f"Error loading substitutions database: {e}")

    # 6. Load barcode database for Packaged Food Scanner
    if os.path.exists(BARCODE_DB_PATH):
        try:
            with open(BARCODE_DB_PATH, 'r', encoding='utf-8') as f:
                barcode_db = json.load(f)
            print(f"Loaded barcode database with {len(barcode_db)} packaged items.")
        except Exception as e:
            print(f"Error loading barcode database: {e}")

# Load resources at startup
load_resources()

# Helper: Compute IoU for NMS
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

# Helper: OpenCV food contour segmentation
def segment_food_item(img_bgr):
    h, w, _ = img_bgr.shape
    total_area_px = w * h

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest)
        x, y, cw, ch = cv2.boundingRect(largest)
        box = [x, y, x + cw, y + ch]

        if contour_area < total_area_px * 0.03 or contour_area > total_area_px * 0.97:
            contour_area = total_area_px * 0.40
            box = [int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)]
    else:
        contour_area = total_area_px * 0.40
        box = [int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)]

    return contour_area, box

# Helper: portion weight estimator
def estimate_weight(class_name, img_shape, contour_area_px):
    h, w, _ = img_shape
    cm_per_px = 25.0 / min(w, h)
    detected_area = contour_area_px * (cm_per_px ** 2)

    anchor_key = "fallback"
    for key in FOOD_ANCHORS:
        if key in class_name.lower():
            anchor_key = key
            break

    anchor = FOOD_ANCHORS[anchor_key]
    ref_weight = anchor["weight"]
    ref_area = anchor["area"]

    estimated = ref_weight * (detected_area / ref_area)
    estimated = float(np.clip(estimated, ref_weight * 0.5, ref_weight * 2.5))
    return round(estimated, 1)

HEALTHIER_ALTERNATIVES = {
    "biryani": "Cauliflower Veggie Biryani / Brown Rice Vegetable Pulav",
    "gulab jamun": "Squeezed Rasgulla / Baked Apples with Cinnamon",
    "samosa": "Baked Veg Samosa / Roasted Seasoned Chickpeas",
    "pizza": "Cauliflower Crust Veggie Pizza / Whole Wheat Pita Veg Pizza",
    "naan": "Tandoori Whole Wheat Roti / Multigrain Chapati",
    "burger": "Portobello Mushroom Burger / Black Bean Veggie Burger",
    "fries": "Baked Sweet Potato Wedges / Roasted Zucchini Sticks",
    "butter chicken": "Tofu Butter Masala / Soya Chaap Makhani",
    "chicken": "Paneer Tikka / Soya Malai Chaap",
    "paneer butter masala": "Tofu Kadai / Low-fat Palak Paneer",
    "rasgulla": "Stevia-sweetened Fruit Compote",
    "jalebi": "Baked Apple Crisps",
    "chole bhature": "Chole with Whole Wheat Roti (No Bhatura)",
    "pakora": "Air-fried Veg Bhaji / Roasted Edamame",
}

RECIPES_DB = {
    "biryani": [
        {"name": "Cauliflower Rice Biryani", "calories": "210 kcal", "ingredients": "Cauliflower rice, lean chicken, low-fat yogurt, fresh mint, coriander, spices."},
        {"name": "Brown Rice Chicken Biryani", "calories": "380 kcal", "ingredients": "Brown basmati rice, chicken breast, onion, spices, 1 tsp ghee."},
        {"name": "Quinoa Veggie Biryani", "calories": "290 kcal", "ingredients": "Quinoa, green peas, carrots, low-fat paneer, spices."}
    ],
    "paneer": [
        {"name": "Grilled Paneer Tikka", "calories": "240 kcal", "ingredients": "Low-fat paneer cubes, bell peppers, onions, skim-milk yogurt marinade, spices."},
        {"name": "Low-fat Palak Paneer", "calories": "180 kcal", "ingredients": "Paneer cubes, spinach puree, garlic, ginger, green chilies, light seasoning."},
        {"name": "Paneer Bhurji (Scrambled)", "calories": "210 kcal", "ingredients": "Scrambled paneer, tomatoes, onions, green peas, minimal oil."}
    ],
    "dal": [
        {"name": "Yellow Moong Dal (Tadka)", "calories": "140 kcal", "ingredients": "Split yellow moong dal, tomato, cumin seeds, garlic, turmeric, fresh cilantro."},
        {"name": "Masoor Dal Curry", "calories": "160 kcal", "ingredients": "Whole red lentils, onion, ginger-garlic paste, tomatoes, curry spices."},
        {"name": "Sprouted Moong Salad", "calories": "110 kcal", "ingredients": "Sprouted moong beans, cucumber, onion, tomato, lemon juice, chaat masala."}
    ],
    "salad": [
        {"name": "Sprout & Veggie Mix", "calories": "120 kcal", "ingredients": "Mixed sprouts, chopped cucumbers, tomatoes, bell peppers, lemon-coriander dressing."},
        {"name": "Mediterranean Chickpea Salad", "calories": "220 kcal", "ingredients": "Boiled chickpeas, cucumber, cherry tomatoes, olives, parsley, 1 tsp olive oil."},
        {"name": "High-Protein Paneer Salad", "calories": "190 kcal", "ingredients": "Low-fat paneer cubes, lettuce, cucumber, tomatoes, lemon-mint vinaigrette."}
    ],
    "chicken": [
        {"name": "Tandoori Chicken Breast", "calories": "220 kcal", "ingredients": "Lean chicken breast, Greek yogurt, tandoori masala, lemon juice, grilled."},
        {"name": "Chicken Clear Soup", "calories": "130 kcal", "ingredients": "Chicken shreds, cabbage, carrots, onion, garlic broth, black pepper."},
        {"name": "Air-Fried Chicken Kabab", "calories": "190 kcal", "ingredients": "Minced chicken, mint, coriander, spices, air-fried."}
    ]
}

def estimate_freshness(img_bgr):
    """
    Analyzes visual freshness of food based on color metrics (HSV).
    Detects if food looks overcooked/charred (extremely dark/low value),
    undercooked/pale (extremely light/low saturation), or optimal/fresh.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {"status": "Undetermined", "desc": "No image region to analyze.", "css_class": "bg-secondary text-light"}
        
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    mean_v = np.mean(v)
    mean_s = np.mean(s)
    
    if mean_v < 48:
        return {
            "status": "Overcooked / Charred",
            "desc": "The food region has very low brightness, suggesting it may be charred, over-fried, or burnt.",
            "css_class": "bg-danger text-light"
        }
    elif mean_v > 205 and mean_s < 30:
        return {
            "status": "Undercooked / Pale",
            "desc": "The food region has high brightness and very low color saturation, suggesting it may be raw, undercooked, or unseasoned.",
            "css_class": "bg-warning text-dark"
        }
    else:
        return {
            "status": "Fresh / Optimal",
            "desc": "The food displays optimal color vibrancy and balanced brightness, indicating it is freshly cooked and ready to eat.",
            "css_class": "bg-success text-light"
        }

# Helper: Nutrition Lookup and calculation
def calculate_nutrition(class_name, weight):
    key = class_name.lower().strip()
    
    nut_info = nutrition_db.get(key, {})
    if not nut_info:
        for db_key, db_val in nutrition_db.items():
            if db_key in key or key in db_key:
                nut_info = db_val
                break
                
    if not nut_info:
        # Fallback values
        nut_info = {
            "display_name": class_name.title(),
            "calories_per_100g": 150.0,
            "carbs_per_100g": 15.0,
            "protein_per_100g": 5.0,
            "fat_per_100g": 8.0,
            "serving_size_info": "1 portion (100g)"
        }
        
    factor = weight / 100.0
    
    # Calculate values
    calories = round(nut_info.get("calories_per_100g", 150) * factor, 1)
    carbs = round(nut_info.get("carbs_per_100g", 15) * factor, 1)
    protein = round(nut_info.get("protein_per_100g", 5) * factor, 1)
    fat = round(nut_info.get("fat_per_100g", 8) * factor, 1)
    
    # Synthesize additional values (Fiber, Sugar, Sodium, Category, Health Rating)
    # Check if they exist or fallback to realistic percentages
    fiber = round(nut_info.get("fiber_per_100g", max(0.5, carbs * 0.12)) * factor, 1)
    sugar = round(nut_info.get("sugar_per_100g", max(0.2, carbs * 0.08)) * factor, 1)
    sodium = round(nut_info.get("sodium_per_100g", 250.0) * factor, 1)
    
    # Get Category
    category = "Snacks"
    for key_anchor, val in FOOD_ANCHORS.items():
        if key_anchor in key:
            category = val["category"]
            break
            
    # Calculate Health Rating score (1.0 to 5.0 scale)
    base_score = 4.5
    
    # Penalize high calorie density, high fat, and high sugar
    if nut_info.get("calories_per_100g", 150) > 350:
        base_score -= 1.5
    elif nut_info.get("calories_per_100g", 150) > 220:
        base_score -= 0.8
        
    if nut_info.get("fat_per_100g", 8) > 20:
        base_score -= 1.2
    elif nut_info.get("fat_per_100g", 8) > 10:
        base_score -= 0.6
        
    if sugar > 15 * factor:
        base_score -= 1.0
    elif sugar > 5 * factor:
        base_score -= 0.4
        
    # Reward high protein and fiber
    if nut_info.get("protein_per_100g", 5) > 12:
        base_score += 0.8
    elif nut_info.get("protein_per_100g", 5) > 6:
        base_score += 0.4
        
    if nut_info.get("fiber_per_100g", 2) > 4:
        base_score += 0.5
        
    # Cap score
    base_score = max(1.0, min(5.0, base_score))
    
    # Map to letter rating
    if base_score >= 4.5:
        health_rating = "A+"
    elif base_score >= 4.0:
        health_rating = "A"
    elif base_score >= 3.3:
        health_rating = "B"
    elif base_score >= 2.6:
        health_rating = "C"
    elif base_score >= 1.8:
        health_rating = "D"
    else:
        health_rating = "E"
        
    # Glycemic Index (GI) calculation
    gi = nut_info.get("glycemic_index", 55)
    if gi <= 55:
        gi_category = "Low"
        gi_color = "success"
    elif gi <= 69:
        gi_category = "Medium"
        gi_color = "warning"
    else:
        gi_category = "High"
        gi_color = "danger"

    # Allergens
    allergens = nut_info.get("allergens", [])

    # Calorie Burn Estimator (MET Formula for 70kg user)
    # Walking (MET 3.5), Running (MET 8.0), Cycling (MET 6.0)
    walking_min = int(round(calories / 4.08)) if calories > 0 else 0
    running_min = int(round(calories / 9.33)) if calories > 0 else 0
    cycling_min = int(round(calories / 7.00)) if calories > 0 else 0

    burn_estimates = {
        "walking_min": max(1, walking_min),
        "running_min": max(1, running_min),
        "cycling_min": max(1, cycling_min)
    }

    # Carbon Footprint per Meal (gCO2)
    carbon_per_100g = nut_info.get("carbon_footprint_gco2_per_100g", 100)
    carbon_total = round(carbon_per_100g * factor, 1)
    # Eco comparison: charging a smartphone uses ~8.22g CO2 per charge
    phone_charges = max(1, round(carbon_total / 8.22))
    if carbon_total < 80:
        eco_rating = "Low"
        eco_color = "success"
    elif carbon_total < 200:
        eco_rating = "Moderate"
        eco_color = "warning"
    else:
        eco_rating = "High"
        eco_color = "danger"
    eco_comparison = f"This meal produced ~{int(carbon_total)}g CO2 — like charging your phone {phone_charges} time{'s' if phone_charges != 1 else ''}."

    # Cultural Context
    cultural_context = nut_info.get("cultural_context", {
        "origin": "India",
        "traditional_pairing": "Rice or Roti",
        "festival": "Traditional Indian cuisine",
        "fun_fact": "A delicious traditional Indian dish enjoyed across the country."
    })

    return {
        "display_name": nut_info.get("display_name", class_name.title().replace("_", " ")),
        "calories": calories,
        "carbs": carbs,
        "protein": protein,
        "fat": fat,
        "fiber": fiber,
        "sugar": sugar,
        "sodium": sodium,
        "serving_size": nut_info.get("serving_size_info", f"1 portion ({int(weight)}g)"),
        "weight": weight,
        "category": category,
        "health_rating": health_rating,
        "glycemic_index": gi,
        "gi_category": gi_category,
        "gi_color": gi_color,
        "allergens": allergens,
        "burn_estimates": burn_estimates,
        "carbon_footprint_g": carbon_total,
        "carbon_per_100g": carbon_per_100g,
        "eco_rating": eco_rating,
        "eco_color": eco_color,
        "eco_comparison": eco_comparison,
        "cultural_context": cultural_context
    }

# ----------------- FLASK PAGE ROUTING -----------------

@app.route('/')
def home():
    device_info = "GPU" if torch.cuda.is_available() else "CPU"
    return render_template('home.html', device=device_info, class_count=len(class_names))

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/nutrition')
def nutrition_page():
    return render_template('nutrition.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/contact')
def contact_page():
    return render_template('contact.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

# ----------------- FLASK API ROUTING -----------------

def merge_close_boxes(detections, img_width, img_height):
    """
    Merge bounding boxes that are close to each other (belong to the same meal/plate).
    Returns a list of merged bounding boxes: [[xmin, ymin, xmax, ymax], ...]
    """
    if not detections:
        return []
        
    boxes = [d["box"] for d in detections]
    
    # We will merge boxes iteratively
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = [False] * len(boxes)
        
        for i in range(len(boxes)):
            if used[i]:
                continue
            
            curr_box = boxes[i]
            used[i] = True
            
            # Find any other box close to curr_box
            for j in range(len(boxes)):
                if used[j]:
                    continue
                
                other_box = boxes[j]
                
                # Check overlap (intersection of boxes)
                overlap = (max(curr_box[0], other_box[0]) < min(curr_box[2], other_box[2])) and \
                          (max(curr_box[1], other_box[1]) < min(curr_box[3], other_box[3]))
                          
                if overlap:
                    close = True
                else:
                    # Calculate minimal distance between box edges
                    dx = max(0, other_box[0] - curr_box[2], curr_box[0] - other_box[2])
                    dy = max(0, other_box[1] - curr_box[3], curr_box[1] - other_box[3])
                    dist = (dx**2 + dy**2)**0.5
                    
                    max_dim = max(img_width, img_height)
                    close = dist < (0.28 * max_dim) # 28% of max image dimension
                
                if close:
                    # Merge boxes
                    curr_box = [
                        min(curr_box[0], other_box[0]),
                        min(curr_box[1], other_box[1]),
                        max(curr_box[2], other_box[2]),
                        max(curr_box[3], other_box[3])
                    ]
                    used[j] = True
                    merged = True
            
            new_boxes.append(curr_box)
        
        boxes = new_boxes
        
    return boxes

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global pytorch_model, yolo_model, idx_to_class, nutrition_db
    
    if pytorch_model is None:
        return jsonify({"success": False, "error_type": "model_not_loaded", "error": "AI classification model is not loaded. Check server logs."}), 500
        
    if 'file' not in request.files:
        return jsonify({"success": False, "error_type": "no_image", "error": "No image file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error_type": "no_image", "error": "No image selected"}), 400
        
    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.jpg', '.png', '.jpeg', '.webp']:
        return jsonify({"success": False, "error_type": "invalid_image", "error": "Invalid file format. Allowed: JPG, PNG, JPEG, WEBP."}), 400
        
    try:
        start_time = time.time()
        
        # Save image
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Check size (10 MB)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if file_size_mb > 10.0:
            os.remove(filepath)
            return jsonify({"success": False, "error_type": "invalid_image", "error": f"Image file is too large ({file_size_mb:.2f} MB). Maximum size is 10 MB."}), 400
            
        # Read image with OpenCV
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            os.remove(filepath)
            return jsonify({"success": False, "error_type": "invalid_image", "error": "Corrupted or invalid image data."}), 400
            
        img_height, img_width, _ = img_bgr.shape
        pil_image = Image.open(filepath).convert('RGB')
        
        # 1. Primary Classification: Always run EfficientNet-B3 on full image first
        top_k = 5
        full_preds = model_predict(pil_image, pytorch_model, idx_to_class, top_k=top_k)
        
        if not full_preds:
            os.remove(filepath)
            return jsonify({"success": False, "error_type": "prediction_failed", "error": "AI prediction did not return results."}), 500

        best_predictions = full_preds
        top_pred = best_predictions[0]
        food_class = top_pred["class"]
        confidence = top_pred["confidence"]
        
        display_image_url = f"/static/uploads/{filename}"
        crop_box = [0, 0, img_width, img_height]
        used_yolo = False

        # 2. Optional Bounding Box Detection with YOLOv8 for COCO food items
        if yolo_model is not None:
            results = yolo_model(filepath, conf=0.25, verbose=False)
            boxes = results[0].boxes
            
            raw_dets = []
            for box in boxes:
                coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                
                # Filter specific COCO food items (excluding generic containers like 'bowl')
                if cls_idx in FOOD_COCO_CLASSES:
                    raw_dets.append({
                        "box": coords,
                        "confidence": conf,
                        "class_name": FOOD_COCO_CLASSES[cls_idx]
                    })
            
            # Apply basic NMS to remove duplicates
            raw_dets = sorted(raw_dets, key=lambda x: x['confidence'], reverse=True)
            yolo_detections = []
            for det in raw_dets:
                overlap = False
                for f_det in yolo_detections:
                    if compute_iou(det['box'], f_det['box']) > 0.45:
                        overlap = True
                        break
                if not overlap:
                    yolo_detections.append(det)
                    
            merged_boxes = merge_close_boxes(yolo_detections, img_width, img_height)
            
            # If YOLO found 1 distinct food box, test if cropped prediction yields high confidence
            if len(merged_boxes) == 1:
                test_crop_box = [int(x) for x in merged_boxes[0]]
                crop_w = test_crop_box[2] - test_crop_box[0]
                crop_h = test_crop_box[3] - test_crop_box[1]
                # Ensure crop box is not tiny (< 15% of total image area)
                if (crop_w * crop_h) >= (0.15 * img_width * img_height):
                    pad_w = int(crop_w * 0.05)
                    pad_h = int(crop_h * 0.05)
                    xmin = max(0, test_crop_box[0] - pad_w)
                    ymin = max(0, test_crop_box[1] - pad_h)
                    xmax = min(img_width, test_crop_box[2] + pad_w)
                    ymax = min(img_height, test_crop_box[3] + pad_h)
                    crop_pil = pil_image.crop((xmin, ymin, xmax, ymax))
                    
                    crop_preds = model_predict(crop_pil, pytorch_model, idx_to_class, top_k=top_k)
                    
                    # Use crop ONLY if crop prediction confidence is high (>= 0.35)
                    if crop_preds and crop_preds[0]["confidence"] >= 0.35:
                        best_predictions = crop_preds
                        top_pred = best_predictions[0]
                        food_class = top_pred["class"]
                        confidence = top_pred["confidence"]
                        crop_box = test_crop_box
                        used_yolo = True
                        
                        annotated_filename = f"annotated_{filename}"
                        annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
                        annotated_img = img_bgr.copy()
                        cv2.rectangle(annotated_img, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (0, 220, 0), 3)
                        cv2.putText(annotated_img, f"Detected: {food_class.title()}", (crop_box[0], max(crop_box[1] - 10, 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
                        cv2.imwrite(annotated_path, annotated_img)
                        display_image_url = f"/static/uploads/{annotated_filename}"

        print(f"[PREDICT] Final Food Class: '{food_class}' | Confidence: {confidence*100:.2f}% | Used BBox: {used_yolo}")

        # Rejection rule: Only reject if confidence is lower than min threshold (0.15)
        confidence_threshold = session.get('threshold', 0.15)
        if confidence < confidence_threshold:
            os.remove(filepath)
            return jsonify({
                "success": False,
                "error_type": "no_food",
                "error": "No recognizable food item detected in the image."
            }), 200

        # Segment crop to estimate portion weight
        crop_bgr = img_bgr[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] if used_yolo else img_bgr
        contour_area, _ = segment_food_item(crop_bgr)
        weight = estimate_weight(food_class, crop_bgr.shape, contour_area)
        
        # Calculate full nutritional breakdown
        nutri_details = calculate_nutrition(food_class, weight)
        
        # Calculate timing & device
        end_time = time.time()
        inference_time = round((end_time - start_time) * 1000, 1) # ms
        device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        if session.get('device') == 'CPU':
            device_name = "CPU"
            
        # Date & Time for logs
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Log to Database and capture entry_id
        entry_id = history_db.add_prediction(
            image_path=display_image_url,
            food_name=nutri_details["display_name"],
            calories=nutri_details["calories"],
            confidence=round(confidence * 100, 2),
            date=date_str,
            time=time_str,
            protein=nutri_details["protein"],
            carbs=nutri_details["carbs"],
            fat=nutri_details["fat"],
            weight=weight
        )
        
        # Map predictions list for Top-5
        top_5_mapped = []
        for rank, p in enumerate(best_predictions, 1):
            class_key = p["class"].lower()
            name = nutrition_db.get(class_key, {}).get("display_name", p["class"].title().replace("_", " "))
            top_5_mapped.append({
                "rank": rank,
                "food_name": name,
                "confidence": round(p["confidence"] * 100, 2),
                "class_key": class_key
            })

        # Get specific recipe suggestions
        recipes = []
        for r_key, r_val in RECIPES_DB.items():
            if r_key in food_class.lower():
                recipes = r_val
                break
        if not recipes:
            recipes = RECIPES_DB.get("salad")

        # Get healthier alternative
        alternative = "Fresh green salad or steamed vegetables"
        for a_key, a_val in HEALTHIER_ALTERNATIVES.items():
            if a_key in food_class.lower():
                alternative = a_val
                break

        # Meal Timing Intelligence
        current_hour = now.hour
        meal_timing_alert = None
        if (current_hour >= 21 or current_hour < 5) and (nutri_details["carbs"] > 25.0 or nutri_details["calories"] > 300):
            meal_timing_alert = "🌙 Late Night High-Carb Notice: Consuming high-carb or heavy foods after 9:00 PM may slow digestion and elevate overnight blood sugar levels."

        # High-Sodium Hydration Alert (Feature 12)
        sodium_hydration_alert = None
        if nutri_details.get("sodium", 0) > 500:
            sodium_hydration_alert = f"💧 High Sodium Hydration Alert: This dish contains {int(nutri_details['sodium'])}mg sodium. Drink 2 extra glasses of water (~500ml) today to help flush excess sodium!"

        return jsonify({
            "success": True,
            "entry_id": entry_id,
            "image_url": display_image_url,
            "food_name": nutri_details["display_name"],
            "food_class": food_class,
            "confidence": round(confidence * 100, 2),
            "inference_time_ms": inference_time,
            "device": device_name,
            "model_version": "v1.2 (EfficientNet-B3)",
            "nutrition": nutri_details,
            "top_5": top_5_mapped,
            "freshness": estimate_freshness(crop_bgr),
            "requires_confirmation": confidence < 0.50,
            "healthier_alternative": alternative,
            "recipes": recipes,
            "meal_timing_alert": meal_timing_alert,
            "sodium_hydration_alert": sodium_hydration_alert
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error_type": "server_error", "error": f"Internal prediction engine error: {str(e)}"}), 500

@app.route('/api/history', methods=['GET'])
def api_get_history():
    search_q = request.args.get('q', None)
    try:
        history_list = history_db.get_predictions(search_q)
        return jsonify({"success": True, "history": history_list})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/history/delete/<int:entry_id>', methods=['DELETE'])
def api_delete_history(entry_id):
    try:
        history_db.delete_prediction(entry_id)
        return jsonify({"success": True, "message": f"Record {entry_id} deleted."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    try:
        history_db.clear_history()
        return jsonify({"success": True, "message": "Prediction history database cleared successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/history/export', methods=['GET'])
def api_export_history():
    try:
        csv_data = history_db.export_to_csv()
        response = app.response_class(
            response=csv_data,
            status=200,
            mimetype='text/csv'
        )
        response.headers.set('Content-Disposition', 'attachment', filename='nutrilens_prediction_history.csv')
        return response
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def api_get_metrics():
    metrics_path = 'model/test_metrics.json'
    history_path = 'model/training_history.json'
    
    metrics = {
        "accuracy": 0.8267,
        "precision": 0.8493,
        "recall": 0.8267,
        "f1_score": 0.8291,
        "classes_count": len(class_names)
    }
    
    history_data = []
    
    # Read metrics file if exists
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics["accuracy"] = data.get("accuracy", 0.8267)
                metrics["precision"] = data.get("precision", 0.8493)
                metrics["recall"] = data.get("recall", 0.8267)
                metrics["f1_score"] = data.get("f1_score", 0.8291)
        except Exception as e:
            print(f"Error reading metrics JSON: {e}")
            
    # Read training history if exists
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
        except Exception as e:
            print(f"Error reading history JSON: {e}")
            
    return jsonify({
        "success": True,
        "metrics": metrics,
        "history": history_data
    })

@app.route('/api/nutrition/all', methods=['GET'])
def api_get_all_nutrition():
    # Return list of all foods available in nutrition database
    results = []
    for key, val in nutrition_db.items():
        results.append({
            "class_name": key,
            "display_name": val.get("display_name", key.title()),
            "calories_per_100g": val.get("calories_per_100g", 150),
            "carbs_per_100g": val.get("carbs_per_100g", 15),
            "protein_per_100g": val.get("protein_per_100g", 5),
            "fat_per_100g": val.get("fat_per_100g", 8),
            "serving_size_info": val.get("serving_size_info", "1 portion (100g)")
        })
    results = sorted(results, key=lambda x: x["display_name"])
    return jsonify({"success": True, "foods": results})

@app.route('/api/nutrition/search', methods=['GET'])
def api_nutrition_search():
    """Search and filter foods by nutrient criteria. Supports:
       ?q=samosa (text search)
       ?high_protein=true (protein > 10g per 100g)
       ?low_carb=true (carbs < 15g per 100g)
       ?low_calorie=true (calories < 150 per 100g)
       ?max_calories=200
       ?min_protein=8
       ?low_gi=true (glycemic index <= 55)
    """
    q = request.args.get('q', '').lower().strip()
    high_protein = request.args.get('high_protein', '').lower() == 'true'
    low_carb = request.args.get('low_carb', '').lower() == 'true'
    low_calorie = request.args.get('low_calorie', '').lower() == 'true'
    low_gi = request.args.get('low_gi', '').lower() == 'true'
    max_calories = request.args.get('max_calories', type=float, default=None)
    min_protein = request.args.get('min_protein', type=float, default=None)

    results = []
    for key, val in nutrition_db.items():
        if key == "unknown":
            continue
        cal = val.get("calories_per_100g", 150)
        prot = val.get("protein_per_100g", 5)
        carb = val.get("carbs_per_100g", 15)
        gi = val.get("glycemic_index", 55)
        display = val.get("display_name", key.title())

        # Text search filter
        if q and q not in key and q not in display.lower():
            continue
        # Nutrient filters
        if high_protein and prot <= 10:
            continue
        if low_carb and carb >= 15:
            continue
        if low_calorie and cal >= 150:
            continue
        if low_gi and gi > 55:
            continue
        if max_calories is not None and cal > max_calories:
            continue
        if min_protein is not None and prot < min_protein:
            continue

        results.append({
            "class_name": key,
            "display_name": display,
            "calories_per_100g": cal,
            "protein_per_100g": prot,
            "carbs_per_100g": carb,
            "fat_per_100g": val.get("fat_per_100g", 8),
            "glycemic_index": gi,
            "serving_size_info": val.get("serving_size_info", "1 portion (100g)")
        })

    results = sorted(results, key=lambda x: x["display_name"])
    return jsonify({"success": True, "count": len(results), "foods": results})

@app.route('/api/report/pdf/<entry_id>', methods=['GET'])
def api_report_pdf(entry_id):
    """Generate a printable PDF nutrition report for a specific scan entry."""
    try:
        entry = history_db.get_prediction_by_id(entry_id)
        if not entry:
            return jsonify({"success": False, "error": "Entry not found."}), 404

        food_name = entry.get("food_name", "Unknown Food")
        calories = entry.get("calories", 0)
        protein = entry.get("protein", 0)
        carbs = entry.get("carbs", 0)
        fat = entry.get("fat", 0)
        date_str = entry.get("date", "Unknown")
        time_str = entry.get("time", "")

        # Try to use reportlab for PDF generation
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import cm
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from io import BytesIO
            from flask import send_file

            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=22, textColor=HexColor('#1a73e8'), spaceAfter=12)
            subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=11, textColor=HexColor('#555555'), spaceAfter=20)
            
            elements = []
            elements.append(Paragraph("Nutrifexa AI - Nutrition Report", title_style))
            elements.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style))
            elements.append(Spacer(1, 12))
            
            # Food info table
            info_data = [
                ["Food Item", food_name],
                ["Scan Date", f"{date_str} {time_str}"],
                ["Entry ID", str(entry_id)],
            ]
            info_table = Table(info_data, colWidths=[4*cm, 12*cm])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), HexColor('#f0f4f8')),
                ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#333333')),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            elements.append(info_table)
            elements.append(Spacer(1, 20))
            
            # Nutrition Facts table
            elements.append(Paragraph("Nutrition Facts", styles['Heading2']))
            nutri_data = [
                ["Nutrient", "Amount", "% Daily Value*"],
                ["Calories", f"{calories:.0f} kcal", f"{(calories/2000*100):.0f}%"],
                ["Protein", f"{protein:.1f} g", f"{(protein/50*100):.0f}%"],
                ["Carbohydrates", f"{carbs:.1f} g", f"{(carbs/300*100):.0f}%"],
                ["Fat", f"{fat:.1f} g", f"{(fat/65*100):.0f}%"],
            ]
            nutri_table = Table(nutri_data, colWidths=[6*cm, 5*cm, 5*cm])
            nutri_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a73e8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dddddd')),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f8f9fa')]),
            ]))
            elements.append(nutri_table)
            elements.append(Spacer(1, 20))
            
            # Footer
            elements.append(Paragraph("* % Daily Values are based on a 2,000 calorie diet.", styles['Normal']))
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Report generated by Nutrifexa AI - AI-Powered Food Recognition & Nutrition Estimation System", subtitle_style))
            
            doc.build(elements)
            buffer.seek(0)
            
            safe_name = food_name.replace(" ", "_").lower()
            return send_file(buffer, mimetype='application/pdf', as_attachment=True, download_name=f"nutrilens_report_{safe_name}_{entry_id}.pdf")

        except ImportError:
            # Fallback: Generate an HTML printable report
            html_content = f"""<!DOCTYPE html>
<html><head><title>Nutrifexa AI Report - {food_name}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; color: #333; }}
h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
.info {{ background: #f0f4f8; padding: 15px; border-radius: 8px; margin: 15px 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th {{ background: #1a73e8; color: white; padding: 12px; text-align: left; }}
td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
.footer {{ color: #888; font-size: 12px; margin-top: 30px; border-top: 1px solid #ddd; padding-top: 10px; }}
@media print {{ body {{ margin: 0; }} }}
</style></head><body>
<h1>Nutrifexa AI - Nutrition Report</h1>
<p style="color:#666;">Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
<div class="info">
<strong>Food Item:</strong> {food_name}<br>
<strong>Scan Date:</strong> {date_str} {time_str}<br>
<strong>Entry ID:</strong> {entry_id}
</div>
<h2>Nutrition Facts</h2>
<table>
<tr><th>Nutrient</th><th>Amount</th><th>% Daily Value*</th></tr>
<tr><td>Calories</td><td>{calories:.0f} kcal</td><td>{(calories/2000*100):.0f}%</td></tr>
<tr><td>Protein</td><td>{protein:.1f} g</td><td>{(protein/50*100):.0f}%</td></tr>
<tr><td>Carbohydrates</td><td>{carbs:.1f} g</td><td>{(carbs/300*100):.0f}%</td></tr>
<tr><td>Fat</td><td>{fat:.1f} g</td><td>{(fat/65*100):.0f}%</td></tr>
</table>
<p class="footer">* % Daily Values are based on a 2,000 calorie diet.<br>
Report generated by Nutrifexa AI - AI-Powered Food Recognition & Nutrition Estimation System</p>
<script>window.print();</script>
</body></html>"""
            return html_content, 200, {'Content-Type': 'text/html'}
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/mutate', methods=['POST'])
def api_mutate():
    """AI Plate Mutator — returns original vs diet-modified nutrition comparison."""
    global nutrition_db, substitutions_db
    
    data = request.json or {}
    food_class = data.get('food_class', '').lower().strip()
    diet = data.get('diet', '').lower().strip()
    weight = float(data.get('weight', 100.0))
    
    valid_diets = ['keto', 'diabetic', 'high_protein', 'low_calorie', 'vegan']
    if diet not in valid_diets:
        return jsonify({"success": False, "error": f"Invalid diet. Choose from: {', '.join(valid_diets)}"}), 400
    
    if not food_class:
        return jsonify({"success": False, "error": "No food class provided."}), 400
    
    # 1. Get original nutrition
    nut_info = nutrition_db.get(food_class, {})
    if not nut_info:
        for db_key, db_val in nutrition_db.items():
            if db_key in food_class or food_class in db_key:
                nut_info = db_val
                food_class = db_key
                break
    
    if not nut_info:
        nut_info = {
            "display_name": food_class.title(),
            "calories_per_100g": 150.0,
            "carbs_per_100g": 15.0,
            "protein_per_100g": 5.0,
            "fat_per_100g": 8.0
        }
    
    factor = weight / 100.0
    original = {
        "calories": round(nut_info.get("calories_per_100g", 150) * factor, 1),
        "carbs": round(nut_info.get("carbs_per_100g", 15) * factor, 1),
        "protein": round(nut_info.get("protein_per_100g", 5) * factor, 1),
        "fat": round(nut_info.get("fat_per_100g", 8) * factor, 1)
    }
    
    # 2. Get food category from substitutions_db
    food_to_cat = substitutions_db.get("food_to_category", {})
    category = food_to_cat.get(food_class, "snack")
    
    # 3. Get category-level multipliers
    cat_defaults = substitutions_db.get("category_defaults", {})
    cat_diet = cat_defaults.get(category, cat_defaults.get("snack", {})).get(diet, {})
    
    cal_mult = cat_diet.get("cal_mult", 1.0)
    carb_mult = cat_diet.get("carb_mult", 1.0)
    protein_mult = cat_diet.get("protein_mult", 1.0)
    fat_mult = cat_diet.get("fat_mult", 1.0)
    recipe_tip = cat_diet.get("tip", "Try healthier cooking methods for this dish.")
    
    # 4. Compute mutated nutrition
    mutated = {
        "calories": round(original["calories"] * cal_mult, 1),
        "carbs": round(original["carbs"] * carb_mult, 1),
        "protein": round(original["protein"] * protein_mult, 1),
        "fat": round(original["fat"] * fat_mult, 1)
    }
    
    # 5. Compute deltas
    delta = {
        "calories": round(mutated["calories"] - original["calories"], 1),
        "carbs": round(mutated["carbs"] - original["carbs"], 1),
        "protein": round(mutated["protein"] - original["protein"], 1),
        "fat": round(mutated["fat"] - original["fat"], 1)
    }
    
    # 6. Get specific ingredient-level swaps if available
    specific = substitutions_db.get("specific_swaps", {}).get(food_class, {}).get(diet, {})
    swaps = specific.get("swaps", [])
    
    # 7. Get diet metadata
    diet_meta = substitutions_db.get("diet_metadata", {}).get(diet, {})
    
    return jsonify({
        "success": True,
        "food_class": food_class,
        "display_name": nut_info.get("display_name", food_class.title()),
        "diet": diet,
        "diet_label": diet_meta.get("label", diet.title()),
        "diet_icon": diet_meta.get("icon", "fa-utensils"),
        "diet_color": diet_meta.get("color", "#3b82f6"),
        "category": category,
        "weight": weight,
        "original": original,
        "mutated": mutated,
        "delta": delta,
        "swaps": swaps,
        "recipe_tip": recipe_tip
    })

@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'POST':
        data = request.json or {}
        if 'device' in data:
            session['device'] = data['device']
        if 'threshold' in data:
            try:
                session['threshold'] = float(data['threshold'])
            except ValueError:
                pass
        if 'calorie_goal' in data:
            try:
                session['calorie_goal'] = float(data['calorie_goal'])
            except ValueError:
                pass
        if 'allergies' in data:
            session['allergies'] = list(data['allergies'])
        return jsonify({"success": True, "settings": {
            "device": session.get('device', 'GPU' if torch.cuda.is_available() else 'CPU'),
            "threshold": session.get('threshold', 0.25),
            "calorie_goal": session.get('calorie_goal', 2000),
            "allergies": session.get('allergies', [])
        }})
    else:
        return jsonify({
            "success": True,
            "settings": {
                "device": session.get('device', 'GPU' if torch.cuda.is_available() else 'CPU'),
                "threshold": session.get('threshold', 0.25),
                "calorie_goal": session.get('calorie_goal', 2000),
                "allergies": session.get('allergies', [])
            }
        })

# ----------------- NEW API ENDPOINTS -----------------

@app.route('/api/history/today', methods=['GET'])
def api_todays_totals():
    """Returns the sum of all macros consumed today plus the user's calorie goal."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    try:
        totals = history_db.get_todays_totals(today)
        goal = session.get('calorie_goal', 2000)
        totals["calorie_goal"] = goal
        totals["remaining"] = round(goal - totals["calories"], 1)
        
        # Smart diet warnings
        warnings = []
        if totals["calories"] >= goal:
            warnings.append({"level": "danger", "icon": "fa-fire", "msg": f"You've exceeded your daily calorie goal of {int(goal)} kcal!"})
        elif totals["remaining"] <= 200 and totals["calories"] > 0:
            warnings.append({"level": "warning", "icon": "fa-exclamation-triangle", "msg": f"Only {totals['remaining']:.0f} kcal remaining for today — eat light!"})
        
        if totals["fat"] > 65:
            warnings.append({"level": "warning", "icon": "fa-droplet", "msg": f"High fat intake today ({totals['fat']:.1f}g). Recommended daily limit is ~65g."})
        if totals["protein"] < 15 and totals["calories"] > 500:
            warnings.append({"level": "info", "icon": "fa-dumbbell", "msg": f"Low protein intake so far ({totals['protein']:.1f}g). Add dal, paneer, eggs or sprouts to your next meal."})
        
        # Nutrient Deficit Alerts (Feature 8)
        if totals["calories"] > 300:
            # Only show deficit alerts if user has eaten enough to evaluate
            if totals.get("protein", 0) < 30:
                warnings.append({"level": "info", "icon": "fa-dumbbell", "msg": f"Protein Deficit: You have eaten only {totals['protein']:.0f}g protein today (recommended: 50g). Add dal, paneer, chicken, or sprouts to your next meal."})
            if totals.get("fiber", 0) < 12:
                warnings.append({"level": "info", "icon": "fa-leaf", "msg": f"Fiber Deficit: Only {totals.get('fiber', 0):.0f}g fiber consumed today (recommended: 25g). Include salad, fruits, or whole grains."})
            if totals.get("carbs", 0) > 250:
                warnings.append({"level": "warning", "icon": "fa-wheat-awn", "msg": f"Carb Overload: {totals['carbs']:.0f}g carbohydrates consumed — well above the recommended 250g daily limit."})
        
        totals["warnings"] = warnings
        return jsonify({"success": True, "totals": totals})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/predict/confirm', methods=['POST'])
def api_confirm_prediction():
    """Corrects a prediction label via second-opinion selection."""
    global nutrition_db
    data = request.json or {}
    entry_id = data.get('entry_id')
    new_class_key = data.get('class_key', '').lower().strip()
    weight = float(data.get('weight', 100.0))
    
    if not entry_id or not new_class_key:
        return jsonify({"success": False, "error": "Missing entry_id or class_key."}), 400
    
    try:
        nutri = calculate_nutrition(new_class_key, weight)
        history_db.update_prediction_name(
            entry_id, nutri["display_name"],
            nutri["calories"], nutri["protein"], nutri["carbs"], nutri["fat"]
        )
        return jsonify({"success": True, "updated_nutrition": nutri})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/bmi/calculate', methods=['POST'])
def api_bmi_calculate():
    """Calculates BMI and daily calorie target using the Harris-Benedict equation."""
    data = request.json or {}
    try:
        age = float(data.get('age', 25))
        weight_kg = float(data.get('weight', 70))
        height_cm = float(data.get('height', 170))
        gender = data.get('gender', 'male').lower()
        activity = data.get('activity', 'moderate').lower()
        
        # BMI
        height_m = height_cm / 100.0
        bmi = round(weight_kg / (height_m ** 2), 1)
        
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        
        # BMR (Harris-Benedict)
        if gender == 'female':
            bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        else:
            bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
        
        # Activity multiplier
        multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        mult = multipliers.get(activity, 1.55)
        daily_calories = round(bmr * mult, 0)
        
        # Save to session
        session['calorie_goal'] = daily_calories
        
        return jsonify({
            "success": True,
            "bmi": bmi,
            "bmi_category": bmi_category,
            "bmr": round(bmr, 1),
            "daily_calories": daily_calories,
            "activity_level": activity
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/nutrition/compare', methods=['POST'])
def api_compare_foods():
    """Compares two foods side-by-side from the nutrition database."""
    global nutrition_db
    data = request.json or {}
    food_a = data.get('food_a', '').lower().strip()
    food_b = data.get('food_b', '').lower().strip()
    
    if not food_a or not food_b:
        return jsonify({"success": False, "error": "Please provide both food_a and food_b."}), 400
    
    def lookup(key):
        info = nutrition_db.get(key, {})
        if not info:
            for db_key, db_val in nutrition_db.items():
                if db_key in key or key in db_key:
                    info = db_val
                    key = db_key
                    break
        if not info:
            return None
        return {
            "name": info.get("display_name", key.title().replace("_", " ")),
            "calories": info.get("calories_per_100g", 0),
            "carbs": info.get("carbs_per_100g", 0),
            "protein": info.get("protein_per_100g", 0),
            "fat": info.get("fat_per_100g", 0),
            "fiber": info.get("fiber_per_100g", 0),
            "sugar": info.get("sugar_per_100g", 0),
            "sodium": info.get("sodium_per_100g", 0),
            "serving_size": info.get("serving_size_info", "1 portion (100g)")
        }
    
    result_a = lookup(food_a)
    result_b = lookup(food_b)
    
    if not result_a:
        return jsonify({"success": False, "error": f"Food '{food_a}' not found in database."}), 404
    if not result_b:
        return jsonify({"success": False, "error": f"Food '{food_b}' not found in database."}), 404
    
    # Determine winners per metric
    winners = {}
    for metric in ["calories", "fat", "sugar", "sodium"]:
        winners[metric] = "A" if result_a[metric] <= result_b[metric] else "B"
    for metric in ["protein", "fiber"]:
        winners[metric] = "A" if result_a[metric] >= result_b[metric] else "B"
    
    return jsonify({
        "success": True,
        "food_a": result_a,
        "food_b": result_b,
        "winners": winners
    })

@app.route('/report')
def report_page():
    """Renders a printable nutrition report page."""
    return render_template('report.html')

@app.route('/api/history/report', methods=['GET'])
def api_history_report():
    """Returns data for a weekly nutrition summary report."""
    try:
        today = datetime.datetime.now()
        daily_data = []
        for i in range(7):
            d = today - datetime.timedelta(days=i)
            date_str = d.strftime("%Y-%m-%d")
            totals = history_db.get_todays_totals(date_str)
            totals["date"] = date_str
            totals["day_name"] = d.strftime("%A")
            daily_data.append(totals)
        
        daily_data.reverse()
        
        # Aggregate summary
        total_cal = sum(d["calories"] for d in daily_data)
        total_protein = sum(d["protein"] for d in daily_data)
        total_carbs = sum(d["carbs"] for d in daily_data)
        total_fat = sum(d["fat"] for d in daily_data)
        avg_cal = round(total_cal / 7, 1)
        
        # Most eaten foods this week
        all_history = history_db.get_predictions()
        week_start = (today - datetime.timedelta(days=6)).strftime("%Y-%m-%d")
        week_foods = {}
        for entry in all_history:
            if entry.get("date", "") >= week_start:
                name = entry.get("food_name", "Unknown")
                week_foods[name] = week_foods.get(name, 0) + 1
        
        most_eaten = sorted(week_foods.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Best / worst day
        best_day = min(daily_data, key=lambda d: d["calories"]) if any(d["calories"] > 0 for d in daily_data) else daily_data[0]
        worst_day = max(daily_data, key=lambda d: d["calories"])
        
        return jsonify({
            "success": True,
            "daily": daily_data,
            "summary": {
                "total_calories": round(total_cal, 1),
                "total_protein": round(total_protein, 1),
                "total_carbs": round(total_carbs, 1),
                "total_fat": round(total_fat, 1),
                "avg_daily_calories": avg_cal,
                "most_eaten": [{"name": n, "count": c} for n, c in most_eaten],
                "best_day": {"date": best_day["date"], "day": best_day.get("day_name", ""), "calories": best_day["calories"]},
                "worst_day": {"date": worst_day["date"], "day": worst_day.get("day_name", ""), "calories": worst_day["calories"]}
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/weight-prediction')
def weight_prediction_page():
    """Renders the weight prediction page."""
    return render_template('prediction.html')

@app.route('/api/profile', methods=['GET', 'POST'])
def api_user_profile():
    """Retrieves or updates the user profile."""
    if request.method == 'POST':
        try:
            data = request.json or {}
            age = int(data.get('age', 25))
            gender = data.get('gender', 'male').lower()
            height = float(data.get('height', 170))
            activity = data.get('activity_level', 'moderate').lower()
            target_weight = float(data.get('target_weight', 70))
            initial_weight = float(data.get('initial_weight', 70))
            
            history_db.update_user_profile(age, gender, height, activity, target_weight, initial_weight)
            
            # Recalculate session calorie goal
            if gender == 'female':
                bmr = 447.593 + (9.247 * initial_weight) + (3.098 * height) - (4.330 * age)
            else:
                bmr = 88.362 + (13.397 * initial_weight) + (4.799 * height) - (5.677 * age)
            
            multipliers = {
                'sedentary': 1.2,
                'light': 1.375,
                'moderate': 1.55,
                'active': 1.725,
                'very_active': 1.9
            }
            mult = multipliers.get(activity, 1.55)
            session['calorie_goal'] = round(bmr * mult, 0)
            
            return jsonify({"success": True, "message": "Profile updated successfully"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400
    else:
        profile = history_db.get_user_profile()
        return jsonify({"success": True, "profile": profile})

@app.route('/api/weight/log', methods=['GET', 'POST'])
def api_weight_log():
    """Retrieves all weight logs or creates a new log."""
    if request.method == 'POST':
        try:
            data = request.json or {}
            date_str = data.get('date')
            weight = float(data.get('weight'))
            if not date_str or not weight:
                return jsonify({"success": False, "error": "Date and weight are required"}), 400
            
            # Simple format verification (YYYY-MM-DD)
            try:
                datetime.datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                return jsonify({"success": False, "error": "Date must be in YYYY-MM-DD format"}), 400
                
            history_db.add_weight_log(date_str, weight)
            return jsonify({"success": True, "message": "Weight logged successfully"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400
    else:
        logs = history_db.get_weight_history()
        return jsonify({"success": True, "logs": logs})

@app.route('/api/weight/log/<int:log_id>', methods=['DELETE'])
def api_weight_delete(log_id):
    """Deletes a weight log by ID."""
    try:
        history_db.delete_weight_log(log_id)
        return jsonify({"success": True, "message": "Weight log deleted successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/weight/predict', methods=['GET'])
def api_weight_predict():
    """Calculates weekly eating patterns and predicts weight gain/loss over 30 days."""
    try:
        profile = history_db.get_user_profile()
        if not profile:
            return jsonify({"success": True, "has_profile": False})
        
        age = profile['age']
        gender = profile['gender']
        height_cm = profile['height']
        activity = profile['activity_level']
        target_weight = profile['target_weight']
        initial_weight = profile['initial_weight']
        
        # Determine current weight from latest log, fallback to initial weight
        logs = history_db.get_weight_history()
        current_weight = initial_weight
        if logs:
            current_weight = logs[-1]['weight']
            
        # Calculate BMR and TDEE based on current weight
        if gender == 'female':
            bmr = 447.593 + (9.247 * current_weight) + (3.098 * height_cm) - (4.330 * age)
        else:
            bmr = 88.362 + (13.397 * current_weight) + (4.799 * height_cm) - (5.677 * age)
            
        multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        tdee = bmr * multipliers.get(activity, 1.55)
        
        # Get historical food consumption
        food_history = history_db.get_predictions()
        daily_intake = {}
        for entry in food_history:
            d_str = entry.get("date")
            cals = entry.get("calories", 0.0) or 0.0
            if d_str:
                daily_intake[d_str] = daily_intake.get(d_str, 0.0) + cals
                
        # Calculate average calorie intake for the last 7 days
        today = datetime.datetime.now()
        last_7_days_intakes = []
        for i in range(7):
            day_str = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            last_7_days_intakes.append(daily_intake.get(day_str, tdee)) # Fallback to TDEE (0 surplus) if not logged
            
        avg_intake = sum(last_7_days_intakes) / 7.0
        avg_surplus = avg_intake - tdee
        
        # PHYSIOLOGICAL CONSTANT: 1 kg of fat = 7700 kcal
        beta_phys = 1.0 / 7700.0  # kg per kcal
        
        beta_calibrated = beta_phys
        is_calibrated = False
        
        # Calibrate using ML regression if enough weight logs exist (>= 5 logs)
        if len(logs) >= 5:
            # We construct a training set
            # Find cumulative surplus since the date of first log
            logs_sorted = sorted(logs, key=lambda x: x['date'])
            first_log_date = datetime.datetime.strptime(logs_sorted[0]['date'], "%Y-%m-%d")
            
            cumulative_surpluses = []
            weight_deltas = []
            
            for log in logs_sorted[1:]:
                log_date = datetime.datetime.strptime(log['date'], "%Y-%m-%d")
                delta_w = log['weight'] - logs_sorted[0]['weight']
                
                # Sum daily surpluses between first_log_date and log_date
                total_surplus = 0.0
                days_diff = (log_date - first_log_date).days
                for d_offset in range(1, days_diff + 1):
                    day_d = first_log_date + datetime.timedelta(days=d_offset)
                    day_d_str = day_d.strftime("%Y-%m-%d")
                    # If food logged, use actual surplus. Otherwise fallback to 0 (eating at maintenance TDEE)
                    day_intake = daily_intake.get(day_d_str, tdee)
                    total_surplus += (day_intake - tdee)
                
                cumulative_surpluses.append(total_surplus)
                weight_deltas.append(delta_w)
                
            # Perform Linear Regression if we have variations in cumulative surplus
            if len(cumulative_surpluses) >= 4 and np.var(cumulative_surpluses) > 1000:
                try:
                    # delta_W = slope * cumulative_surplus + intercept
                    slope, intercept = np.polyfit(cumulative_surpluses, weight_deltas, 1)
                    
                    # Apply regularization towards the physiological baseline
                    # Weight of empirical data scales from 0 to 1 as logs grow from 5 to 30
                    alpha = min(1.0, (len(logs) - 5) / 25.0)
                    beta_calibrated = alpha * slope + (1.0 - alpha) * beta_phys
                    
                    # Sanity clip to prevent unrealistic metabolic rates
                    # 1 kg = between 4000 and 15000 kcal
                    min_beta = 1.0 / 15000.0
                    max_beta = 1.0 / 4000.0
                    beta_calibrated = np.clip(beta_calibrated, min_beta, max_beta)
                    is_calibrated = True
                except Exception as e:
                    print(f"Regression error: {e}")
                    
        # Future 30-day weight projection
        forecast = []
        for t in range(1, 31):
            pred_date = (today + datetime.timedelta(days=t)).strftime("%Y-%m-%d")
            pred_weight = current_weight + (beta_calibrated * avg_surplus * t)
            forecast.append({
                "date": pred_date,
                "weight": round(float(pred_weight), 2)
            })
            
        # Target calculation: days until reaching target
        days_to_target = None
        if avg_surplus != 0:
            weight_diff = target_weight - current_weight
            # weight_diff = beta_calibrated * avg_surplus * days
            required_calories = weight_diff / beta_calibrated
            # if surplus matches the direction of weight diff
            if (weight_diff > 0 and avg_surplus > 0) or (weight_diff < 0 and avg_surplus < 0):
                days_to_target = int(np.ceil(required_calories / avg_surplus))
                
        return jsonify({
            "success": True,
            "has_profile": True,
            "profile": profile,
            "current_weight": current_weight,
            "target_weight": target_weight,
            "tdee": round(tdee, 1),
            "bmr": round(bmr, 1),
            "avg_daily_intake": round(avg_intake, 1),
            "avg_daily_surplus": round(avg_surplus, 1),
            "is_calibrated": is_calibrated,
            "beta_calibrated_kcal_per_kg": round(float(1.0 / beta_calibrated), 1),
            "days_to_target": days_to_target,
            "history": logs,
            "forecast": forecast
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/community')
def community_feed_page():
    """Renders the community social feed page."""
    return render_template('community.html')

@app.route('/api/community/posts', methods=['GET', 'POST'])
def api_community_posts():
    """Handles fetching and submitting community feed posts."""
    if request.method == 'POST':
        try:
            data = request.json or {}
            username = data.get('username', 'Anonymous').strip()
            food_name = data.get('food_name', 'Healthy Meal')
            calories = float(data.get('calories', 0))
            protein = float(data.get('protein', 0))
            carbs = float(data.get('carbs', 0))
            fat = float(data.get('fat', 0))
            health_rating = data.get('health_rating', 'B')
            image_path = data.get('image_path', '')
            recipe_title = data.get('recipe_title', '').strip()
            recipe_instructions = data.get('recipe_instructions', '').strip()
            
            if not username:
                username = 'Anonymous'
                
            history_db.add_community_post(
                username=username,
                food_name=food_name,
                calories=calories,
                protein=protein,
                carbs=carbs,
                fat=fat,
                health_rating=health_rating,
                image_path=image_path,
                recipe_title=recipe_title,
                recipe_instructions=recipe_instructions
            )
            return jsonify({"success": True, "message": "Meal shared to community feed successfully"})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 400
    else:
        try:
            posts = history_db.get_community_posts()
            return jsonify({"success": True, "posts": posts})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/community/posts/<int:post_id>/like', methods=['POST'])
def api_like_post(post_id):
    """Increments the likes count of a post by 1."""
    try:
        history_db.like_community_post(post_id)
        return jsonify({"success": True, "message": "Liked post successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ----------------- FEATURE 11: BARCODE SCANNER API -----------------
@app.route('/api/barcode/<barcode_code>', methods=['GET'])
def api_barcode_lookup(barcode_code):
    """Looks up packaged food item by barcode (EAN/UPC)."""
    code = str(barcode_code).strip()
    item = barcode_db.get(code)
    if not item:
        # Search by partial match if exact fails
        for b_code, b_data in barcode_db.items():
            if code in b_code:
                item = b_data
                break

    if item:
        return jsonify({"success": True, "found": True, "product": item})
    else:
        return jsonify({
            "success": True,
            "found": False,
            "message": f"Barcode '{code}' not found in database. You can add custom packaged item details or use camera scan."
        }), 4404 if False else 200

# ----------------- FEATURE 12: WATER INTAKE TRACKER API -----------------
@app.route('/api/water/today', methods=['GET'])
def api_water_today():
    """Gets today's water intake in glasses (1 glass = 250ml) and target."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    glasses = history_db.get_todays_water(today)
    target_glasses = 8  # 2000 ml
    return jsonify({
        "success": True,
        "date": today,
        "glasses": glasses,
        "target_glasses": target_glasses,
        "volume_ml": glasses * 250,
        "target_volume_ml": target_glasses * 250,
        "percentage": min(100, int((glasses / target_glasses) * 100))
    })

@app.route('/api/water/log', methods=['POST'])
def api_water_log():
    """Logs water intake (+1 or -1 glass)."""
    data = request.json or {}
    delta = int(data.get('delta', 1))
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    new_total = history_db.log_water(today, delta)
    target = 8
    return jsonify({
        "success": True,
        "glasses": new_total,
        "target_glasses": target,
        "volume_ml": new_total * 250,
        "percentage": min(100, int((new_total / target) * 100))
    })

# ----------------- FEATURE 13: SMART MEAL PLANNER API -----------------
@app.route('/api/meal-planner/recommend', methods=['GET', 'POST'])
def api_meal_planner():
    """Generates a balanced Breakfast, Lunch, and Dinner meal plan based on remaining daily calories."""
    if request.method == 'POST':
        data = request.json or {}
        target_calories = float(data.get('target_calories', 2000))
    else:
        target_calories = float(request.args.get('target_calories', 2000))

    # Calculate meal splits: Breakfast 25%, Lunch 40%, Dinner 35%
    b_cal = target_calories * 0.25
    l_cal = target_calories * 0.40
    d_cal = target_calories * 0.35

    breakfast_options = []
    lunch_options = []
    dinner_options = []

    for key, val in nutrition_db.items():
        if key == "unknown":
            continue
        display = val.get("display_name", key.title())
        cal = val.get("calories_per_100g", 150)
        prot = val.get("protein_per_100g", 5)
        carb = val.get("carbs_per_100g", 15)
        fat = val.get("fat_per_100g", 8)
        serving = val.get("serving_size_info", "1 portion (100g)")

        # Categorize
        if any(w in key for w in ["poha", "idly", "dosa", "upma", "paratha", "oats", "appam", "chilla", "khakhra"]):
            breakfast_options.append({"name": display, "calories": cal, "protein": prot, "carbs": carb, "serving": serving})
        elif any(w in key for w in ["biryani", "rice", "dal", "thali", "curry", "paneer", "chicken", "rajma", "chole"]):
            lunch_options.append({"name": display, "calories": cal, "protein": prot, "carbs": carb, "serving": serving})
            dinner_options.append({"name": display, "calories": cal, "protein": prot, "carbs": carb, "serving": serving})
        elif cal < 200:
            breakfast_options.append({"name": display, "calories": cal, "protein": prot, "carbs": carb, "serving": serving})
        else:
            dinner_options.append({"name": display, "calories": cal, "protein": prot, "carbs": carb, "serving": serving})

    # Pick top 3 for each meal
    b_recs = sorted(breakfast_options, key=lambda x: abs(x["calories"] - b_cal))[:3]
    l_recs = sorted(lunch_options, key=lambda x: abs(x["calories"] - l_cal))[:3]
    d_recs = sorted(dinner_options, key=lambda x: abs(x["calories"] - d_cal))[:3]

    return jsonify({
        "success": True,
        "target_calories": target_calories,
        "meal_splits": {
            "breakfast_target_kcal": round(b_cal),
            "lunch_target_kcal": round(l_cal),
            "dinner_target_kcal": round(d_cal)
        },
        "plan": {
            "breakfast": b_recs,
            "lunch": l_recs,
            "dinner": d_recs
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
