"""
=============================================================================
FILE: predict.py  —  Nutrifexa-AI  (PyTorch / EfficientNet-B3)
=============================================================================
FIX APPLIED:
  ImportError: cannot import name 'load_trained_model' from 'model_utils'
  → Replaced with correct PyTorch import:
      from model_utils import load_model, predict

WHAT THIS SCRIPT DOES:
  1. Loads your trained EfficientNet-B3 PyTorch model
  2. Runs inference on a single food image
  3. Segments the food using OpenCV to estimate weight
  4. Looks up nutrition from nutrition_db.json
  5. Prints a formatted Nutrition Facts label

HOW TO RUN:
  python predict.py --image "C:\path\to\food.jpg"
  python predict.py --image food.jpg --nutrition_db nutrition_db.json
  python predict.py --image food.jpg --top_k 3
=============================================================================
"""

import os
import json
import argparse
import numpy as np
import cv2
from PIL import Image

# ✅ CORRECT PyTorch imports — matches your current model_utils.py exactly
from model_utils import load_model, predict as model_predict

# ─────────────────────────────────────────────────────────────────────────────
# FOOD WEIGHT ANCHORS
# ─────────────────────────────────────────────────────────────────────────────
FOOD_ANCHORS = {
    "dal":          {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "curry":        {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "sambar":       {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "paneer":       {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "chicken":      {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "roti":         {"weight": 40.0,  "area": 176.7, "type": "flatbread"},
    "chapati":      {"weight": 40.0,  "area": 176.7, "type": "flatbread"},
    "naan":         {"weight": 60.0,  "area": 200.0, "type": "flatbread"},
    "paratha":      {"weight": 80.0,  "area": 176.7, "type": "flatbread"},
    "dosa":         {"weight": 120.0, "area": 314.2, "type": "flatbread"},
    "puri":         {"weight": 40.0,  "area": 78.5,  "type": "flatbread"},
    "papad":        {"weight": 15.0,  "area": 176.7, "type": "flatbread"},
    "biryani":      {"weight": 180.0, "area": 113.1, "type": "solid"},
    "rice":         {"weight": 180.0, "area": 113.1, "type": "solid"},
    "chawal":       {"weight": 180.0, "area": 113.1, "type": "solid"},
    "samosa":       {"weight": 100.0, "area": 36.0,  "type": "solid"},
    "vada":         {"weight": 100.0, "area": 50.2,  "type": "solid"},
    "kebab":        {"weight": 120.0, "area": 80.0,  "type": "solid"},
    "gulab jamun":  {"weight": 50.0,  "area": 19.6,  "type": "solid"},
    "rasmalai":     {"weight": 50.0,  "area": 19.6,  "type": "solid"},
    "fallback":     {"weight": 100.0, "area": 100.0, "type": "solid"}
}


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Nutrifexa-AI — Predict food and show nutrition"
    )
    parser.add_argument(
        '--image', required=True,
        help="Path to the input food image (jpg/png/webp)"
    )
    parser.add_argument(
        '--weights_path', default='model/efficientnet_b3_nutrilens.pth',
        help="Path to trained .pth weights (default: model/efficientnet_b3_nutrilens.pth)"
    )
    parser.add_argument(
        '--class_map', default='model/class_indices.json',
        help="Path to class_indices.json (default: model/class_indices.json)"
    )
    parser.add_argument(
        '--nutrition_db', default='nutrition_db.json',
        help="Path to nutrition database JSON (default: nutrition_db.json)"
    )
    parser.add_argument(
        '--top_k', type=int, default=3,
        help="Show top-K predictions (default: 3)"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# NUTRITION DB LOADER
# ─────────────────────────────────────────────────────────────────────────────
def load_nutrition_db(db_path: str) -> dict:
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            return json.load(f)
    print(f"  Warning: Nutrition DB not found at '{db_path}'. Using fallback values.")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# FOOD SEGMENTATION (OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
def segment_food_item(img_bgr: np.ndarray):
    """
    Segments the food item from the background using Otsu thresholding.
    Returns (contour_area_px, bounding_box).
    """
    h, w, _ = img_bgr.shape
    total_area_px = w * h

    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest       = max(contours, key=cv2.contourArea)
        contour_area  = cv2.contourArea(largest)
        x, y, cw, ch = cv2.boundingRect(largest)
        box           = [x, y, x + cw, y + ch]

        # Reject if too small (noise) or too large (entire frame edge)
        if contour_area < total_area_px * 0.03 or contour_area > total_area_px * 0.97:
            contour_area = total_area_px * 0.40
            box = [int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)]
    else:
        contour_area = total_area_px * 0.40
        box = [int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85)]

    return contour_area, box


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────
def estimate_weight(class_name: str, img_shape: tuple,
                    contour_area_px: float, box: list) -> float:
    """
    Estimates serving weight in grams using contour area and food-type anchors.
    Assumes the image represents a ~25cm wide field of view (standard plate closeup).
    """
    h, w, _ = img_shape
    cm_per_px       = 25.0 / min(w, h)
    detected_area   = contour_area_px * (cm_per_px ** 2)

    anchor_key = "fallback"
    for key in FOOD_ANCHORS:
        if key in class_name.lower():
            anchor_key = key
            break

    anchor     = FOOD_ANCHORS[anchor_key]
    ref_weight = anchor["weight"]
    ref_area   = anchor["area"]

    estimated  = ref_weight * (detected_area / ref_area)
    estimated  = float(np.clip(estimated, ref_weight * 0.5, ref_weight * 2.5))
    return round(estimated, 1)


# ─────────────────────────────────────────────────────────────────────────────
# NUTRITION LOOKUP
# ─────────────────────────────────────────────────────────────────────────────
def lookup_nutrition(class_name: str, nutrition_db: dict) -> dict:
    """Exact match → partial match → fallback defaults."""
    key = class_name.lower().strip()

    if key in nutrition_db:
        return nutrition_db[key]

    for db_key, db_val in nutrition_db.items():
        if db_key in key or key in db_key:
            print(f"  Mapped '{class_name}' → '{db_key}' in nutrition DB")
            return db_val

    print(f"  Warning: '{class_name}' not in nutrition DB. Using fallback values.")
    return {
        "display_name":      class_name.title(),
        "calories_per_100g": 150.0,
        "carbs_per_100g":    15.0,
        "protein_per_100g":  5.0,
        "fat_per_100g":      8.0,
        "fiber_per_100g":    2.0,
        "sugar_per_100g":    3.0,
        "sodium_per_100g":   200.0,
        "serving_size_g":    100.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NUTRITION LABEL PRINTER
# ─────────────────────────────────────────────────────────────────────────────
def print_nutrition_label(class_name, confidence, weight,
                          calories, carbs, protein, fat,
                          fiber=None, sugar=None, sodium=None):
    """Prints a clean ASCII Nutrition Facts label to the terminal."""
    title = class_name.upper().replace("_", " ")
    conf_str = f"{confidence*100:.1f}%"

    print(f"""
  ╔══════════════════════════════════════════════╗
  ║           NUTRITION FACTS                    ║
  ╠══════════════════════════════════════════════╣
  ║  Food Item   : {title:<30}║
  ║  Confidence  : {conf_str:<30}║
  ║  Est. Weight : {str(weight)+' g':<30}║
  ╠══════════════════════════════════════════════╣
  ║  Amount Per Serving                          ║
  ╠══════════════════════════════════════════════╣
  ║  CALORIES    : {int(calories):<30}║
  ╠══════════════════════════════════════════════╣
  ║                          % Daily Value*      ║
  ║  Total Fat   : {str(fat)+'g':<12}  {int((fat/65)*100):>3}%              ║
  ║  Total Carbs : {str(carbs)+'g':<12}  {int((carbs/300)*100):>3}%              ║
  ║  Protein     : {str(protein)+'g':<12}  {int((protein/50)*100):>3}%              ║""")

    if fiber is not None:
        print(f"  ║  Fiber       : {str(fiber)+'g':<12}  {int((fiber/28)*100):>3}%              ║")
    if sugar is not None:
        print(f"  ║  Sugar       : {str(sugar)+'g':<12}                   ║")
    if sodium is not None:
        print(f"  ║  Sodium      : {str(int(sodium))+'mg':<12}  {int((sodium/2300)*100):>3}%              ║")

    print(f"""  ╠══════════════════════════════════════════════╣
  ║  * Based on 2000 calorie daily diet          ║
  ╚══════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.image):
        print(f"\n  ERROR: Image not found at '{args.image}'")
        return

    if not os.path.exists(args.weights_path):
        print(f"\n  ERROR: Weights not found at '{args.weights_path}'")
        print(f"  Run 'python train.py' first.")
        return

    if not os.path.exists(args.class_map):
        print(f"\n  ERROR: class_indices.json not found at '{args.class_map}'")
        print(f"  It is saved automatically when train.py finishes.")
        return

    print(f"\n  {'='*50}")
    print(f"  Nutrifexa-AI — Predict  (EfficientNet-B3 / PyTorch)")
    print(f"  {'='*50}")
    print(f"  Image    : {args.image}")
    print(f"  Weights  : {args.weights_path}")
    print(f"  {'='*50}\n")

    # ── Load PyTorch model ────────────────────────────────────────────────────
    # ✅ Correct signature: load_model(model_path, class_map_path) → (model, idx_to_class)
    print("  Loading model ...")
    model, idx_to_class = load_model(
        model_path     = args.weights_path,
        class_map_path = args.class_map
    )
    print(f"  Model loaded. Classes: {len(idx_to_class)}")

    # ── Load nutrition database ───────────────────────────────────────────────
    nutrition_db = load_nutrition_db(args.nutrition_db)

    # ── Load image ────────────────────────────────────────────────────────────
    print(f"  Reading image ...")
    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        print(f"\n  ERROR: OpenCV could not read '{args.image}'.")
        print(f"  Make sure the path has no special characters and the file is a valid image.")
        return

    img_shape = img_bgr.shape
    pil_img   = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # ── Run PyTorch inference ─────────────────────────────────────────────────
    # ✅ Uses model_utils.predict() — correct PyTorch function
    print("  Running inference ...")
    results = model_predict(pil_img, model, idx_to_class, top_k=args.top_k)

    if not results:
        print("  ERROR: No predictions returned.")
        return

    # Top prediction
    top_class      = results[0]["class"]
    top_confidence = results[0]["confidence"]

    print(f"\n  Prediction : {top_class.upper().replace('_', ' ')}")
    print(f"  Confidence : {top_confidence*100:.2f}%")

    # Show all top-K
    if args.top_k > 1:
        print(f"\n  Top-{args.top_k} Predictions:")
        print(f"  {'─'*40}")
        for rank, r in enumerate(results, 1):
            label = r['class'].replace('_', ' ').title()
            conf  = r['confidence'] * 100
            bar   = '█' * int(conf / 5)
            print(f"  {rank}. {label:<25} {conf:5.1f}%  {bar}")
        print()

    # ── Segment food and estimate weight ──────────────────────────────────────
    print("  Estimating portion size ...")
    contour_area, box = segment_food_item(img_bgr)
    weight = estimate_weight(top_class, img_shape, contour_area, box)

    # ── Nutrition lookup ──────────────────────────────────────────────────────
    nut_info     = lookup_nutrition(top_class, nutrition_db)
    display_name = nut_info.get("display_name", top_class.title())
    factor       = weight / 100.0

    calories = round(nut_info.get("calories_per_100g", 150) * factor, 1)
    carbs    = round(nut_info.get("carbs_per_100g",    15)  * factor, 1)
    protein  = round(nut_info.get("protein_per_100g",   5)  * factor, 1)
    fat      = round(nut_info.get("fat_per_100g",        8) * factor, 1)
    fiber    = round(nut_info.get("fiber_per_100g",      2) * factor, 1) \
               if "fiber_per_100g" in nut_info else None
    sugar    = round(nut_info.get("sugar_per_100g",      3) * factor, 1) \
               if "sugar_per_100g" in nut_info else None
    sodium   = round(nut_info.get("sodium_per_100g",   200) * factor, 1) \
               if "sodium_per_100g" in nut_info else None

    # ── Print Nutrition Label ─────────────────────────────────────────────────
    print_nutrition_label(
        display_name, top_confidence, weight,
        calories, carbs, protein, fat,
        fiber, sugar, sodium
    )

    # ── Save result to JSON ───────────────────────────────────────────────────
    result_dict = {
        "image":       args.image,
        "food_name":   top_class,
        "confidence":  round(top_confidence * 100, 2),
        "weight_g":    weight,
        "calories":    calories,
        "protein_g":   protein,
        "carbs_g":     carbs,
        "fat_g":       fat,
        "fiber_g":     fiber,
        "sugar_g":     sugar,
        "sodium_mg":   sodium,
        "all_predictions": results
    }
    os.makedirs("model", exist_ok=True)
    out_path = "model/last_prediction.json"
    with open(out_path, "w") as f:
        json.dump(result_dict, f, indent=4)
    print(f"  Result saved → '{out_path}'\n")


if __name__ == '__main__':
    main()