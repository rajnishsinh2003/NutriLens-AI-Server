import numpy as np

# Reference Weight Anchors (grams) and standard surface areas (cm2) for portions
# Standard bowl diameter is 12 cm -> area = pi * 6^2 = 113.1 cm2
# Standard flatbread diameter is 15 cm -> area = pi * 7.5^2 = 176.7 cm2
# Standard samosa size is 6x6 cm -> area = 36 cm2
# Standard sweet piece diameter is 5 cm -> area = pi * 2.5^2 = 19.6 cm2
FOOD_ANCHORS = {
    # liquids / bowl items (base: 150g, standard bowl area: 113.1 cm2)
    "dal curry": {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "palak paneer": {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "butter chicken": {"weight": 150.0, "area": 113.1, "type": "liquid"},
    "sambar": {"weight": 150.0, "area": 113.1, "type": "liquid"},
    
    # flatbreads (base: 40g, standard roti area: 176.7 cm2)
    "roti": {"weight": 40.0, "area": 176.7, "type": "flatbread"},
    "dosa": {"weight": 120.0, "area": 314.2, "type": "flatbread"}, # Dosa is larger (20cm)
    "puri": {"weight": 40.0, "area": 78.5, "type": "flatbread"},  # Puri is smaller (10cm)
    "papad": {"weight": 15.0, "area": 176.7, "type": "flatbread"},
    
    # rice dishes (base: 180g, standard rice bowl area: 113.1 cm2)
    "biryani": {"weight": 180.0, "area": 113.1, "type": "solid"},
    "rajma chawal": {"weight": 180.0, "area": 113.1, "type": "solid"},
    
    # snacks (base: 100g, standard samosa area: 36.0 cm2)
    "samosa": {"weight": 100.0, "area": 36.0, "type": "solid"},
    "vada": {"weight": 100.0, "area": 50.2, "type": "solid"},
    "kebab": {"weight": 120.0, "area": 80.0, "type": "solid"},
    
    # sweets / desserts (base: 50g, standard sweet area: 19.6 cm2)
    "gulab jamun": {"weight": 50.0, "area": 19.6, "type": "solid"},
    "rasmalai": {"weight": 50.0, "area": 19.6, "type": "solid"},
    
    # fallback defaults
    "unknown": {"weight": 100.0, "area": 100.0, "type": "solid"}
}

def calibrate_scale(detections, img_width, img_height):
    """
    Finds the plate bounding box to calibrate the scale (cm/pixel).
    A standard Indian Thali plate is assumed to be 30 cm in diameter.
    
    If no plate is detected, falls back to assuming the plate occupies 80% 
    of the image's minimum dimension.
    """
    plate_box = None
    max_area = 0
    
    for det in detections:
        box = det['box'] # [xmin, ymin, xmax, ymax]
        w_px = box[2] - box[0]
        h_px = box[3] - box[1]
        area = w_px * h_px
        
        # If a large bounding box (>= 35% of the image area) is found
        if area > max_area and area > (img_width * img_height * 0.35):
            max_area = area
            plate_box = box
            
    if plate_box is not None:
        w_px = plate_box[2] - plate_box[0]
        h_px = plate_box[3] - plate_box[1]
        diameter_px = max(w_px, h_px)
        cm_per_pixel = 30.0 / diameter_px
        print(f"Plate detected. Diameter: {diameter_px:.1f}px. Calibrated scale: {cm_per_pixel:.4f} cm/pixel.")
    else:
        min_dim = min(img_width, img_height)
        diameter_px = min_dim * 0.8
        cm_per_pixel = 30.0 / diameter_px
        print(f"No plate detected. Fallback scale: {cm_per_pixel:.4f} cm/pixel.")
        
    return cm_per_pixel

def estimate_food_weight(box, food_profile, cm_per_pixel):
    """
    Estimates the weight of the food item in grams based on bounding box size,
    calibrated scale, and reference weight anchors.
    """
    xmin, ymin, xmax, ymax = box
    w_px = xmax - xmin
    h_px = ymax - ymin
    
    # Convert dimensions to cm
    w_cm = w_px * cm_per_pixel
    h_cm = h_px * cm_per_pixel
    
    # Bbox area is rectangular, apply circular/oval shape factor (0.78) for round items
    food_class = food_profile.get("display_name", "").lower()
    
    # Find matching anchor configuration
    anchor_key = "unknown"
    for key in FOOD_ANCHORS.keys():
        if key in food_class:
            anchor_key = key
            break
            
    anchor = FOOD_ANCHORS[anchor_key]
    ref_weight = anchor["weight"]
    ref_area = anchor["area"]
    food_type = anchor["type"]
    
    # Calculate detected 2D area in cm2
    shape_factor = 0.78 if food_type in ["liquid", "flatbread"] else 1.0
    detected_area_cm2 = w_cm * h_cm * shape_factor
    
    # Proportional weight scaling: weight = ref_weight * (detected_area / ref_area)
    weight_g = ref_weight * (detected_area_cm2 / ref_area)
    
    # Safety clipping to prevent extreme outlier estimates:
    # Restrict weight to between 50% and 250% of standard serving anchor weight
    min_weight = ref_weight * 0.5
    max_weight = ref_weight * 2.5
    weight_g = np.clip(weight_g, min_weight, max_weight)
    
    return round(float(weight_g), 1)
