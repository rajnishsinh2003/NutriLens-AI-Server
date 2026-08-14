/**
 * =============================================================================
 * FILE: main.js  —  NutriLens-AI  (Interactive Frontend Engine)
 * =============================================================================
 * Handles:
 *   - Drag & Drop + File Browser upload
 *   - Webcam capture via MediaDevices API
 *   - Multi-stage loading animation with step transitions
 *   - /api/predict POST and result rendering
 *   - Dark / Light theme toggling with localStorage persistence
 *   - Mobile sidebar drawer interactions
 * =============================================================================
 */

/* ─────────────────────────────────────────────────────────────────────────────
   GLOBAL STATE
   ───────────────────────────────────────────────────────────────────────────── */
let selectedFile = null;
let webcamStream = null;
let webcamActive = false;

// Plate Mutator state
let lastPredictedFoodClass = null;
let lastPredictedWeight = 100;
let lastEntryId = null;
let currentMutatorDiet = null;
let currentScanData = null;

let allFoodsDatabase = [];
let selectedCompareA = null;
let selectedCompareB = null;
let compareBarChartInstance = null;
let compareRadarChartInstance = null;
let activeComparisonView = 'table';

/* ─────────────────────────────────────────────────────────────────────────────
   1. THEME TOGGLING (Dark ↔ Light)
   ───────────────────────────────────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
    const saved = localStorage.getItem("theme-preference");
    if (saved === "light") {
        applyTheme("light");
    } else {
        applyTheme("dark");
    }

    // Cache all food items for interactive search and recommendations
    fetchAllFoods();
});

async function fetchAllFoods() {
    try {
        const response = await fetch('/api/nutrition/all');
        const data = await response.json();
        if (data.success) {
            allFoodsDatabase = data.foods || [];
            console.log(`Cached ${allFoodsDatabase.length} foods from local database.`);
        }
    } catch (err) {
        console.error("Error fetching all foods:", err);
    }
}

function toggleThemeMode() {
    const isCurrentlyDark = document.body.classList.contains("dark-theme");
    applyTheme(isCurrentlyDark ? "light" : "dark");
}

function applyTheme(mode) {
    if (mode === "light") {
        document.body.classList.remove("dark-theme");
        document.body.classList.add("light-theme");
        document.documentElement.setAttribute("data-bs-theme", "light");
        localStorage.setItem("theme-preference", "light");
    } else {
        document.body.classList.remove("light-theme");
        document.body.classList.add("dark-theme");
        document.documentElement.setAttribute("data-bs-theme", "dark");
        localStorage.setItem("theme-preference", "dark");
    }

    // Toggle sun/moon icons in both sidebar and mobile menu
    document.querySelectorAll(".icon-light-mode").forEach(el => {
        el.classList.toggle("d-none", mode === "dark");
    });
    document.querySelectorAll(".icon-dark-mode").forEach(el => {
        el.classList.toggle("d-none", mode === "light");
    });
}

/* ─────────────────────────────────────────────────────────────────────────────
   2. MOBILE SIDEBAR DRAWER
   ───────────────────────────────────────────────────────────────────────────── */
function toggleMobileMenu() {
    const overlay = document.getElementById("mobile-menu-overlay");
    if (overlay) overlay.classList.toggle("open");
}

/* ─────────────────────────────────────────────────────────────────────────────
   3. DRAG & DROP + FILE BROWSER
   ───────────────────────────────────────────────────────────────────────────── */
function triggerFileInput() {
    const input = document.getElementById("image-file-input");
    if (input) input.click();
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    const zone = document.getElementById("drop-zone-container");
    if (zone) zone.classList.add("dragover");
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    const zone = document.getElementById("drop-zone-container");
    if (zone) zone.classList.remove("dragover");
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    const zone = document.getElementById("drop-zone-container");
    if (zone) zone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processSelectedFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processSelectedFile(files[0]);
    }
}

function processSelectedFile(file) {
    // Validate extension
    const allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
        showError("Invalid Image Format", "Only JPG, PNG, JPEG, and WEBP images are supported. Please choose a valid food image file.");
        return;
    }

    // Validate size (10 MB max)
    const maxBytes = 10 * 1024 * 1024;
    if (file.size > maxBytes) {
        showError("Image Too Large", `The file is ${(file.size / 1024 / 1024).toFixed(2)} MB. Maximum allowed size is 10 MB.`);
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function (ev) {
        const previewPanel = document.getElementById("preview-panel");
        const previewImg = document.getElementById("preview-image-elem");
        if (previewPanel && previewImg) {
            previewImg.src = ev.target.result;
            previewPanel.classList.remove("d-none");
        }
    };
    reader.readAsDataURL(file);
}

/* ─────────────────────────────────────────────────────────────────────────────
   4. WEBCAM HANDLING
   ───────────────────────────────────────────────────────────────────────────── */
function setMode(mode) {
    // When switching to camera tab, prepare UI
    if (mode === "camera") {
        resetPreviewSilent();
    } else if (mode === "upload") {
        stopWebcam();
    }
}

function toggleWebcam() {
    if (webcamActive) {
        stopWebcam();
    } else {
        startWebcam();
    }
}

async function startWebcam() {
    const video = document.getElementById("webcam-stream");
    const toggleBtn = document.getElementById("btn-camera-toggle");
    const captureBtn = document.getElementById("btn-camera-capture");
    if (!video) return;

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: 640, height: 480 }
        });
        video.srcObject = webcamStream;
        webcamActive = true;

        if (toggleBtn) {
            toggleBtn.innerHTML = '<i class="fa-solid fa-circle-stop me-2"></i>Stop Feed';
            toggleBtn.classList.remove("btn-outline-success");
            toggleBtn.classList.add("btn-outline-danger");
        }
        if (captureBtn) captureBtn.disabled = false;
    } catch (err) {
        console.error("Webcam error:", err);
        showError("Camera Unavailable", "Could not access webcam. Make sure camera permissions are granted and the device has a camera.");
    }
}

function stopWebcam() {
    const video = document.getElementById("webcam-stream");
    const toggleBtn = document.getElementById("btn-camera-toggle");
    const captureBtn = document.getElementById("btn-camera-capture");

    if (webcamStream) {
        webcamStream.getTracks().forEach(t => t.stop());
        webcamStream = null;
    }
    if (video) video.srcObject = null;
    webcamActive = false;

    if (toggleBtn) {
        toggleBtn.innerHTML = '<i class="fa-solid fa-circle-play me-2"></i>Start Feed';
        toggleBtn.classList.remove("btn-outline-danger");
        toggleBtn.classList.add("btn-outline-success");
    }
    if (captureBtn) captureBtn.disabled = true;
}

function captureWebcamSnapshot() {
    const video = document.getElementById("webcam-stream");
    const canvas = document.getElementById("snapshot-canvas");
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to a file blob
    canvas.toBlob(function (blob) {
        if (!blob) return;
        const capturedFile = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
        selectedFile = capturedFile;

        // Show preview
        const previewPanel = document.getElementById("preview-panel");
        const previewImg = document.getElementById("preview-image-elem");
        if (previewPanel && previewImg) {
            previewImg.src = canvas.toDataURL("image/jpeg");
            previewPanel.classList.remove("d-none");
        }

        // Stop webcam after capture
        stopWebcam();
    }, "image/jpeg", 0.92);
}

/* ─────────────────────────────────────────────────────────────────────────────
   5. PREDICTION PIPELINE (MULTI-STAGE LOADING + API CALL)
   ───────────────────────────────────────────────────────────────────────────── */
function startAIAnalysis() {
    if (!selectedFile) {
        showError("No Image", "Please upload or capture a food image before starting the analysis.");
        return;
    }

    // Show loading card, hide others
    showCard("loading");

    // Step animation timeline
    const steps = ["step-upload", "step-validate", "step-model", "step-preprocess", "step-prediction", "step-nutrition"];
    const labels = [
        "Uploading image to server...",
        "Validating image boundaries...",
        "Loading PyTorch model weights...",
        "Preprocessing color tensor matrix...",
        "Running EfficientNet-B3 inference...",
        "Fetching portion nutrition data..."
    ];

    let currentStep = 0;

    function activateStep(index) {
        if (index >= steps.length) return;

        const stepEl = document.getElementById(steps[index]);
        if (!stepEl) return;

        stepEl.classList.add("active");
        // Show spinner for current
        const spinner = stepEl.querySelector(".spinner");
        if (spinner) spinner.classList.remove("d-none");

        // Update subtitle text
        const subtext = document.getElementById("loader-subtext");
        if (subtext) subtext.innerText = labels[index];
    }

    function completeStep(index) {
        if (index >= steps.length) return;

        const stepEl = document.getElementById(steps[index]);
        if (!stepEl) return;

        stepEl.classList.remove("active");
        stepEl.classList.add("completed");

        const spinner = stepEl.querySelector(".spinner");
        const checked = stepEl.querySelector(".checked");
        if (spinner) spinner.classList.add("d-none");
        if (checked) checked.classList.remove("d-none");
    }

    // Reset all steps first
    steps.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove("active", "completed");
            const sp = el.querySelector(".spinner");
            const ch = el.querySelector(".checked");
            if (sp) sp.classList.add("d-none");
            if (ch) ch.classList.add("d-none");
        }
    });

    // Start step 0
    activateStep(0);

    // Send the file to the API in parallel
    const formData = new FormData();
    formData.append("file", selectedFile);

    const apiPromise = fetch("/api/predict", {
        method: "POST",
        body: formData
    }).then(res => res.json());

    // Animate steps 0-3 over ~2 seconds with delays, then wait for API
    let stepTimer;

    function advanceStep() {
        if (currentStep < 3) {
            completeStep(currentStep);
            currentStep++;
            activateStep(currentStep);
            stepTimer = setTimeout(advanceStep, 450 + Math.random() * 200);
        }
    }

    stepTimer = setTimeout(advanceStep, 400);

    // When API responds, quickly complete remaining steps and show results
    apiPromise.then(data => {
        // Clear any remaining timers
        clearTimeout(stepTimer);

        // Complete all remaining steps quickly
        function finishRemaining(idx) {
            if (idx < steps.length) {
                completeStep(idx);
                if (idx + 1 < steps.length) {
                    activateStep(idx + 1);
                    setTimeout(() => finishRemaining(idx + 1), 250);
                } else {
                    // All done — show result after a short beat
                    setTimeout(() => handlePredictionResponse(data), 400);
                }
            }
        }

        // Start from wherever we left off
        completeStep(currentStep);
        currentStep++;
        if (currentStep < steps.length) {
            activateStep(currentStep);
            setTimeout(() => finishRemaining(currentStep), 250);
        } else {
            setTimeout(() => handlePredictionResponse(data), 400);
        }
    }).catch(err => {
        clearTimeout(stepTimer);
        console.error("Prediction API error:", err);
        showError("Server Error", "The prediction service is unreachable. Please make sure the Flask backend is running.");
    });
}

function handlePredictionResponse(data) {
    if (!data.success) {
        // Map error types to beautiful error cards
        const errorMap = {
            "multiple_items": { title: "Multiple Food Items Detected", icon: "fa-layer-group" },
            "no_food": { title: "No Food Item Detected", icon: "fa-utensils" },
            "no_image": { title: "No Image Uploaded", icon: "fa-image" },
            "invalid_image": { title: "Invalid Image File", icon: "fa-file-excel" },
            "model_not_loaded": { title: "AI Model Not Loaded", icon: "fa-microchip" },
            "prediction_failed": { title: "Prediction Engine Failed", icon: "fa-robot" },
            "low_confidence": { title: "Low Confidence Detection", icon: "fa-chart-line" }
        };

        const errInfo = errorMap[data.error_type] || { title: "Analysis Error", icon: "fa-triangle-exclamation" };
        showError(errInfo.title, data.error || "An unexpected error occurred during food analysis.");
        return;
    }

    currentScanData = data;
    const n = data.nutrition || {};

    // Populate Prediction Card info
    setTextById("res-food-name", n.display_name || data.food_name || "Unknown");
    setTextById("res-category", n.category || "Food");

    // Confidence meter
    const conf = data.confidence || 0;
    setTextById("res-confidence-text", conf.toFixed(2) + "%");
    const confBar = document.getElementById("res-confidence-bar");
    if (confBar) {
        confBar.style.width = conf + "%";
        confBar.className = "confidence-progress-fill";
        if (conf >= 90) confBar.style.backgroundColor = "var(--color-success)";
        else if (conf >= 70) confBar.style.backgroundColor = "var(--color-warning)";
        else confBar.style.backgroundColor = "var(--color-danger)";
    }

    // Image
    const resImg = document.getElementById("res-image");
    if (resImg) resImg.src = data.image_url || "";

    // Large metric pills
    setTextById("res-kcal", (n.calories || 0).toFixed(1) + " kcal");
    setTextById("res-weight", (n.weight || 0).toFixed(1) + " g");

    // Populate CARD 2: Nutrition Overview
    setTextById("overview-cal", (n.calories || 0).toFixed(1) + " kcal");
    setTextById("overview-protein", (n.protein || 0).toFixed(1) + " g");
    setTextById("overview-carbs", (n.carbs || 0).toFixed(1) + " g");
    setTextById("overview-fat", (n.fat || 0).toFixed(1) + " g");
    setTextById("overview-fiber", (n.fiber || 0).toFixed(1) + " g");
    setTextById("overview-sugar", (n.sugar || 0).toFixed(1) + " g");
    setTextById("overview-sodium", Math.round(n.sodium || 0) + " mg");
    setTextById("overview-serving", n.serving_size || "1 portion");

    // FEATURE 1: Glycemic Index (GI) Tracker
    const gi = n.glycemic_index || 55;
    const giCat = n.gi_category || "Medium";
    const giColor = n.gi_color || "warning";

    setTextById("overview-gi", `${gi} (${giCat})`);
    const giBadge = document.getElementById("res-gi-badge");
    if (giBadge) {
        giBadge.textContent = `${giCat} GI (${gi})`;
        giBadge.className = `badge bg-${giColor} text-light`;
    }

    // FEATURE 2: Allergen Alert System
    const foodAllergens = n.allergens || [];
    setTextById("overview-allergens", foodAllergens.length > 0 ? foodAllergens.map(a => a.toUpperCase()).join(", ") : "None Detected");

    const userAllergies = JSON.parse(localStorage.getItem("user_allergies") || "[]");
    const matchedAllergens = foodAllergens.filter(alg => userAllergies.includes(alg.toLowerCase()));

    const allergenBanner = document.getElementById("allergen-alert-banner");
    const allergenBody = document.getElementById("allergen-alert-body");
    if (allergenBanner && allergenBody) {
        if (matchedAllergens.length > 0) {
            allergenBody.innerHTML = `This scanned food item contains <strong>${matchedAllergens.map(a => a.toUpperCase()).join(", ")}</strong> which matches your saved personal allergy preferences!`;
            allergenBanner.classList.remove("d-none");
        } else {
            allergenBanner.classList.add("d-none");
        }
    }

    // FEATURE 3: Calorie Burn Estimator
    const burns = n.burn_estimates || {
        walking_min: Math.max(1, Math.round((n.calories || 0) / 4.08)),
        running_min: Math.max(1, Math.round((n.calories || 0) / 9.33)),
        cycling_min: Math.max(1, Math.round((n.calories || 0) / 7.00))
    };
    setTextById("burn-cal-display", (n.calories || 0).toFixed(0));
    setTextById("burn-walk-min", burns.walking_min + " min");
    setTextById("burn-run-min", burns.running_min + " min");
    setTextById("burn-cycle-min", burns.cycling_min + " min");

    // FEATURE 5: Meal Timing Intelligence Alert
    const timingBanner = document.getElementById("meal-timing-alert-banner");
    const timingBody = document.getElementById("meal-timing-alert-body");
    if (timingBanner && timingBody) {
        if (data.meal_timing_alert) {
            timingBody.textContent = data.meal_timing_alert;
            timingBanner.classList.remove("d-none");
        } else {
            timingBanner.classList.add("d-none");
        }
    }
    // FEATURE 6: Carbon Footprint per Meal
    setTextById("carbon-total-g", (n.carbon_footprint_g || 0).toFixed(0) + "g");
    setTextById("carbon-eco-comparison", n.eco_comparison || `This meal produced ~${(n.carbon_footprint_g || 0).toFixed(0)}g CO2.`);
    const ecoBadge = document.getElementById("carbon-eco-badge");
    if (ecoBadge) {
        ecoBadge.textContent = (n.eco_rating || "Moderate") + " Eco Impact";
        ecoBadge.className = `badge bg-${n.eco_color || 'warning'} text-light`;
    }
    const carbonProgress = document.getElementById("carbon-progress-bar");
    if (carbonProgress) {
        const pct = Math.min(100, Math.max(10, Math.round(((n.carbon_footprint_g || 100) / 350) * 100)));
        carbonProgress.style.width = pct + "%";
        carbonProgress.className = `progress-bar bg-${n.eco_color || 'warning'}`;
    }

    // FEATURE 7: Food Cultural Context Card
    const ctx = n.cultural_context || {};
    setTextById("cultural-origin", ctx.origin || "India");
    setTextById("cultural-pairing", ctx.traditional_pairing || "Rice or Roti");
    setTextById("cultural-festival", ctx.festival || "Traditional");
    setTextById("cultural-funfact", ctx.fun_fact || "A staple traditional Indian dish.");

    // Populate CARD 3: Health Score circular gauge
    const score = clientComputeHealthScore(n, n.weight);
    setTextById("health-gauge-val", score);

    const gaugeStatusEl = document.getElementById("health-gauge-status");
    const gaugeFillEl = document.getElementById("health-gauge-fill");

    let statusText = "Moderate";
    let statusColor = "var(--color-warning)";

    if (score >= 75) {
        statusText = "Healthy";
        statusColor = "var(--color-success)";
    } else if (score < 40) {
        statusText = "Unhealthy";
        statusColor = "var(--color-danger)";
    }

    if (gaugeStatusEl) {
        gaugeStatusEl.textContent = statusText;
        gaugeStatusEl.style.color = statusColor;
    }
    if (gaugeFillEl) {
        // Circumference is 2 * PI * 54 = 339.3
        const circ = 339.3;
        const offset = circ * (1 - (score / 100));
        gaugeFillEl.style.strokeDashoffset = offset;
        gaugeFillEl.style.stroke = statusColor;
    }

    // Populate AI Health Summary explanation text
    const summaryTextEl = document.getElementById("health-summary-text");
    if (summaryTextEl) {
        summaryTextEl.innerHTML = `<strong>${n.display_name || data.food_name}</strong> is rated as <strong>${statusText.toLowerCase()}</strong> with an AI Health Score of <strong>${score}/100</strong>. ` +
            `It contains ${(n.calories || 0).toFixed(0)} kcal and is estimated at ${(n.weight || 0).toFixed(0)}g portion weight. ` +
            `Pair with high-protein and high-fiber side dishes to ensure a balanced nutritional value.`;
    }
    setTextById("health-tip-text", `Tip: Cook with healthy oils and try pairing with yogurt or light veggies salad.`);

    // Populate Alternatives, Coach and comparative badges
    populateAlternativesAndCoach(data);

    // Initialize Comparison Food A with the predicted food (Option 1)
    selectedCompareA = {
        display_name: n.display_name || data.food_name,
        class_name: data.food_class || data.food_name,
        calories_per_100g: n.calories_per_100g || ((n.calories / n.weight) * 100) || 150,
        carbs_per_100g: n.carbs_per_100g || ((n.carbs / n.weight) * 100) || 15,
        protein_per_100g: n.protein_per_100g || ((n.protein / n.weight) * 100) || 5,
        fat_per_100g: n.fat_per_100g || ((n.fat / n.weight) * 100) || 8,
        fiber_per_100g: n.fiber_per_100g || ((n.fiber / n.weight) * 100) || 1,
        sugar_per_100g: n.sugar_per_100g || ((n.sugar / n.weight) * 100) || 1,
        sodium_per_100g: n.sodium_per_100g || ((n.sodium / n.weight) * 100) || 250,
        serving_size_info: n.serving_size || "1 portion"
    };

    // Set A display pill
    document.getElementById("compare-a-name").textContent = selectedCompareA.display_name;
    document.getElementById("compare-a-cal").textContent = (selectedCompareA.calories_per_100g * (n.weight / 100)).toFixed(0) + " kcal";
    document.getElementById("compare-a-search-wrapper").classList.add("d-none");
    document.getElementById("compare-a-display").classList.remove("d-none");

    // Preload Food B as Sandwich or first available DB item to show a starting comparison comparison
    if (allFoodsDatabase.length > 0) {
        let bMatch = allFoodsDatabase.find(f => f.class_name.includes("sandwich") || f.class_name.includes("salad") || f.class_name.includes("roti"));
        if (!bMatch) bMatch = allFoodsDatabase[0];
        selectCompareFood('B', bMatch.class_name);
    }

    // Store food class & weight for Plate Mutator
    lastPredictedFoodClass = data.food_class || data.food_name || n.display_name || "Unknown";
    lastPredictedWeight = n.weight || 100;
    lastEntryId = data.entry_id || null;
    currentMutatorDiet = null;
    resetMutatorUI();

    // Populate AI Insights Panel (if element exists)
    populateInsights(data);

    // Populate Second Opinion panel if low confidence
    populateSecondOpinion(data);

    // Refresh daily tracker
    refreshDailyTracker();

    showCard("results");
}

/* ─────────────────────────────────────────────────────────────────────────────
   6. CLIENT-SIDE AI HEALTH LOGIC & COMPARISON ENGINE
   ───────────────────────────────────────────────────────────────────────────── */

// Compute health score of a food item client-side
function clientComputeHealthScore(n, weight) {
    if (!weight || weight === 0) weight = 100;
    const factor = 100 / weight;

    // Convert portion macros back to per 100g if not already per 100g (which properties in DB are)
    const calories = (n.calories_per_100g !== undefined) ? n.calories_per_100g : (n.calories || 150) * factor;
    const protein = (n.protein_per_100g !== undefined) ? n.protein_per_100g : (n.protein || 5) * factor;
    const carbs = (n.carbs_per_100g !== undefined) ? n.carbs_per_100g : (n.carbs || 15) * factor;
    const fat = (n.fat_per_100g !== undefined) ? n.fat_per_100g : (n.fat || 8) * factor;
    const fiber = (n.fiber_per_100g !== undefined) ? n.fiber_per_100g : (n.fiber || (carbs * 0.12)) * factor;
    const sugar = (n.sugar_per_100g !== undefined) ? n.sugar_per_100g : (n.sugar || (carbs * 0.08)) * factor;
    const sodium = (n.sodium_per_100g !== undefined) ? n.sodium_per_100g : (n.sodium || 250) * factor;

    let score = 50.0;

    if (protein > 15) score += 15;
    else if (protein > 8) score += 8;

    if (fiber > 5) score += 10;
    else if (fiber > 2) score += 5;

    if (calories > 350) score -= 20;
    else if (calories > 220) score -= 10;
    else if (calories < 120) score += 10;

    if (fat > 20) score -= 15;
    else if (fat > 12) score -= 8;

    if (sugar > 15) score -= 12;
    else if (sugar > 8) score -= 6;

    if (sodium > 500) score -= 10;
    else if (sodium > 300) score -= 5;

    return Math.round(Math.min(Math.max(score, 0), 100));
}

// Populate healthier alternatives and diet coach tips
function populateAlternativesAndCoach(data) {
    const n = data.nutrition || {};
    const weight = n.weight || 100;
    const predScore = clientComputeHealthScore(n, weight);
    const predCategory = (n.category || "Snacks").toLowerCase();

    // Score all foods in database
    const scoredFoods = allFoodsDatabase.map(f => {
        return {
            ...f,
            healthScore: clientComputeHealthScore(f, 100)
        };
    });

    const nonVegRegex = /\b(chicken|mutton|fish|egg|eggs|pork|beef|prawn|prawns|shrimp|lamb|meat|crab|seafood)\b/i;
    function isVeg(food) {
        const name = (food.display_name || food.class_name || "").toLowerCase();
        const cat = (food.category || "").toLowerCase();
        return !nonVegRegex.test(name) && !nonVegRegex.test(cat);
    }

    function getDishCategory(foodName, catName) {
        const s = ((foodName || "") + " " + (catName || "")).toLowerCase();
        if (s.match(/\b(dal|curry|sambar|paneer|korma|gravy|rasam|bhaji|sabzi|chole|rajma|aloo gobi|kadai|makhani|palak)\b/)) return "curry";
        if (s.match(/\b(roti|naan|chapati|paratha|bhatura|puri|papad|bread|kulcha|phulka)\b/)) return "bread";
        if (s.match(/\b(biryani|rice|pulao|pulav|khichdi|chawal|fried rice|jeera rice)\b/)) return "rice";
        if (s.match(/\b(samosa|pakora|vada|kachori|tikki|chaat|bhel|dhokla|fries|burger|momos|roll|chilla|cutlet|poha|upma|idli|dosa|uttapam)\b/)) return "snack";
        if (s.match(/\b(jamun|jalebi|rasgulla|rasmalai|halwa|kheer|ladoo|laddu|barfi|sweet|payasam|ice cream|cake|rabri|kulfi|mysore pak)\b/)) return "dessert";
        if (s.match(/\b(lassi|milk|juice|tea|coffee|shake|smoothie|soup|water|buttermilk|chaas)\b/)) return "beverage";
        return "other";
    }

    const targetCat = getDishCategory(data.food_name || data.food_class, n.category);

    // Filter 100% vegetarian foods excluding current scanned item
    let vegFoods = scoredFoods.filter(f => f.class_name !== data.food_class.toLowerCase() && isVeg(f));

    // Split into Same Dish Category vs General Healthy Veg Foods
    let sameCatAlts = vegFoods.filter(f => getDishCategory(f.display_name || f.class_name, f.category) === targetCat && f.healthScore >= predScore);
    sameCatAlts.sort((a, b) => b.healthScore - a.healthScore);

    let otherAlts = vegFoods.filter(f => getDishCategory(f.display_name || f.class_name, f.category) !== targetCat && f.healthScore > predScore);
    otherAlts.sort((a, b) => b.healthScore - a.healthScore);

    // Pick top 3 from same category, fill remainder from top general healthy veg options
    const sameCatPickCount = Math.min(3, sameCatAlts.length);
    let finalAlts = [...sameCatAlts.slice(0, sameCatPickCount), ...otherAlts.slice(0, 4 - sameCatPickCount)];
    if (finalAlts.length < 4) {
        finalAlts = [...sameCatAlts, ...otherAlts].slice(0, 4);
    }

    // Render alternatives row
    const container = document.getElementById("alternatives-row");
    if (container) {
        if (finalAlts.length === 0) {
            container.innerHTML = `<div class="text-secondary small py-2 w-100 text-center">No healthier alternatives found in local database.</div>`;
        } else {
            container.innerHTML = finalAlts.map(alt => {
                // Estimate icon based on category
                let iconClass = "fa-apple-whole";
                const dn = alt.display_name.toLowerCase();
                if (dn.includes("sandwich") || dn.includes("bread") || dn.includes("roti") || dn.includes("naan") || dn.includes("poha")) iconClass = "fa-bread-slice";
                else if (dn.includes("salad") || dn.includes("chaat") || dn.includes("sprouts")) iconClass = "fa-leaf";
                else if (dn.includes("samosa") || dn.includes("vada") || dn.includes("snack") || dn.includes("fries")) iconClass = "fa-cookie";
                else if (dn.includes("soup") || dn.includes("dal") || dn.includes("sambar") || dn.includes("curry") || dn.includes("paneer")) iconClass = "fa-bowl-food";

                return `
                    <div class="alternative-card">
                        <div class="alt-thumbnail">
                            <i class="fa-solid ${iconClass}"></i>
                        </div>
                        <div class="text-light fw-semibold text-truncate small" title="${alt.display_name}">${alt.display_name}</div>
                        <div class="d-flex justify-content-between align-items-center mt-auto">
                            <span class="text-secondary xx-small">${alt.calories_per_100g} kcal</span>
                            <span class="badge bg-success-subtle text-success xx-small" style="font-size: 0.65rem;">Score: ${alt.healthScore}</span>
                        </div>
                        <button type="button" class="btn btn-outline-primary btn-sm py-1 mt-1 xx-small fw-semibold" onclick="selectCompareFoodB('${alt.class_name}')">
                            <i class="fa-solid fa-scale-balanced me-1"></i> Compare
                        </button>
                    </div>
                `;
            }).join("");
        }
    }

    // Why These Are Better Card
    const whyBetterContainer = document.getElementById("why-better-badges");
    if (whyBetterContainer) {
        if (finalAlts.length === 0) {
            whyBetterContainer.innerHTML = `<div class="text-secondary small">Not enough comparison data.</div>`;
        } else {
            const bestAlt = finalAlts[0];
            const predCal100 = n.calories / (weight / 100);
            const predFat100 = n.fat / (weight / 100);
            const predProt100 = n.protein / (weight / 100);
            const predFiber100 = n.fiber / (weight / 100);
            const predSodium100 = n.sodium / (weight / 100);

            const badges = [];

            // Calories
            if (bestAlt.calories_per_100g < predCal100) {
                const diffPct = Math.round(((predCal100 - bestAlt.calories_per_100g) / predCal100) * 100);
                if (diffPct > 5) badges.push(`<span class="why-better-pill calories"><i class="fa-solid fa-arrow-down"></i> ${diffPct}% Lower Calories</span>`);
            }
            // Fat
            if (bestAlt.fat_per_100g < predFat100 && predFat100 > 2) {
                const diffPct = Math.round(((predFat100 - bestAlt.fat_per_100g) / predFat100) * 100);
                if (diffPct > 5) badges.push(`<span class="why-better-pill fat"><i class="fa-solid fa-arrow-down"></i> ${diffPct}% Less Fat</span>`);
            }
            // Protein
            if (bestAlt.protein_per_100g > predProt100) {
                badges.push(`<span class="why-better-pill protein"><i class="fa-solid fa-arrow-up"></i> Higher Protein</span>`);
            }
            // Fiber
            if (bestAlt.fiber_per_100g > predFiber100 || (bestAlt.carbs_per_100g * 0.12) > predFiber100) {
                badges.push(`<span class="why-better-pill fiber"><i class="fa-solid fa-leaf"></i> More Fiber</span>`);
            }
            // Sodium
            if (bestAlt.sodium_per_100g < predSodium100 && predSodium100 > 100) {
                badges.push(`<span class="why-better-pill sodium"><i class="fa-solid fa-shield-halved"></i> Lower Sodium</span>`);
            }

            if (badges.length === 0) {
                badges.push(`<span class="why-better-pill protein"><i class="fa-solid fa-thumbs-up"></i> Better Overall Score</span>`);
            }

            whyBetterContainer.innerHTML = badges.join("");
        }
    }

    // AI Diet Coach Tips
    // Best Time
    let bestTime = "Snack";
    const category = (n.category || "Snacks").toLowerCase();
    const calories = n.calories || 150;
    const protein = n.protein || 5;
    const carbs = n.carbs || 15;

    if (category.includes('dessert') || category.includes('sweet') || (n.sugar && n.sugar > 20)) {
        bestTime = "Snack / Treat";
    } else if (calories < 150 && carbs < 20) {
        bestTime = "Breakfast / Snack";
    } else if (protein > 15 && calories > 200) {
        bestTime = "Dinner / Lunch";
    } else if (calories >= 150 && calories <= 350) {
        bestTime = "Lunch / Breakfast";
    } else {
        bestTime = "Lunch (Main Meal)";
    }
    document.getElementById("coach-best-time").textContent = bestTime;

    // Eat With
    let eatWith = "Fresh salad or light soup";
    if (category.includes('curry') || category.includes('dal')) {
        eatWith = "Whole Wheat Roti / Salad";
    } else if (category.includes('bread') || category.includes('roti')) {
        eatWith = "Dal Curry / Steamed Veggies";
    } else if (category.includes('snack') || category.includes('fry')) {
        eatWith = "Mint Chutney / Yogurt Dip";
    } else if (category.includes('dessert') || category.includes('sweet')) {
        eatWith = "Warm water or limit portion";
    }
    document.getElementById("coach-eat-with").textContent = eatWith;

    // Avoid With
    let avoidWith = "Sugary sodas or extra oils";
    if (calories > 350 || (n.fat && n.fat > 15)) {
        avoidWith = "Fried Side Dishes / Soda";
    } else {
        avoidWith = "Creamy Dressings / Sugary Juice";
    }
    document.getElementById("coach-avoid-with").textContent = avoidWith;

    // Good For
    let goodFor = "Balanced Diet";
    if (protein > 15) {
        goodFor = "Muscle Recovery & Growth";
    } else if (calories < 150 && (n.fat && n.fat < 5)) {
        goodFor = "Weight Loss Management";
    } else if (category.includes('dessert') || category.includes('sweet') || calories > 400) {
        goodFor = "Occasional Indulgence";
    }
    document.getElementById("coach-good-for").textContent = goodFor;
}

// Autocomplete search input handling
function handleSearchInput(type) {
    const inputEl = document.getElementById(type === 'A' ? 'search-compare-a' : 'search-compare-b');
    const dropdownEl = document.getElementById(type === 'A' ? 'search-dropdown-a' : 'search-dropdown-b');
    if (!inputEl || !dropdownEl) return;

    const query = inputEl.value.trim().toLowerCase();
    if (!query) {
        dropdownEl.style.display = 'none';
        return;
    }

    // Filter database
    const matches = allFoodsDatabase.filter(f => f.display_name.toLowerCase().includes(query));

    if (matches.length === 0) {
        dropdownEl.innerHTML = `<div class="p-2 text-secondary small text-center">No matching foods found</div>`;
    } else {
        dropdownEl.innerHTML = matches.slice(0, 5).map(match => `
            <div class="search-item-row" onclick="selectCompareFood('${type}', '${match.class_name}')">
                <div class="d-flex align-items-center gap-2">
                    <i class="fa-solid fa-utensils text-secondary small"></i>
                    <span class="text-light small fw-medium">${match.display_name}</span>
                </div>
                <span class="text-secondary small">${match.calories_per_100g} kcal/100g</span>
            </div>
        `).join("");
    }
    dropdownEl.style.display = 'block';
}

function selectCompareFood(type, classKey) {
    // Hide dropdown
    document.getElementById(type === 'A' ? 'search-dropdown-a' : 'search-dropdown-b').style.display = 'none';

    // Lookup item details
    const food = allFoodsDatabase.find(f => f.class_name === classKey);
    if (!food) return;

    if (type === 'A') {
        selectedCompareA = food;
        document.getElementById("compare-a-name").textContent = food.display_name;
        document.getElementById("compare-a-cal").textContent = food.calories_per_100g + " kcal";

        // Hide search input, show display pill
        document.getElementById("compare-a-search-wrapper").classList.add("d-none");
        document.getElementById("compare-a-display").classList.remove("d-none");
    } else {
        selectedCompareB = food;
        document.getElementById("search-compare-b").value = food.display_name;
    }

    // Automatically trigger preview updates if both foods are loaded
    if (selectedCompareA && selectedCompareB) {
        updateComparisonPreview();
    }
}

function selectCompareFoodB(classKey) {
    selectCompareFood('B', classKey);
    setComparisonView('table');
}

function clearCompareA() {
    selectedCompareA = null;
    document.getElementById("compare-a-display").classList.add("d-none");
    document.getElementById("compare-a-search-wrapper").classList.remove("d-none");
    document.getElementById("search-compare-a").value = "";
    document.getElementById("search-compare-a").focus();
}

function triggerCompareNow() {
    if (!selectedCompareA) {
        alert("Please select or search Food A first!");
        return;
    }
    if (!selectedCompareB) {
        alert("Please select or search Food B first!");
        return;
    }
    updateComparisonPreview();
}

function updateComparisonPreview() {
    if (!selectedCompareA || !selectedCompareB) return;

    const a = selectedCompareA;
    const b = selectedCompareB;

    // Set headers
    document.getElementById("comp-header-a").textContent = a.display_name;
    document.getElementById("comp-header-b").textContent = b.display_name;

    // Calculate metrics
    const metrics = [
        { label: "Calories (kcal)", valA: a.calories_per_100g || a.calories || 0, valB: b.calories_per_100g || b.calories || 0, lowerIsBetter: true },
        { label: "Protein (g)", valA: a.protein_per_100g || a.protein || 0, valB: b.protein_per_100g || b.protein || 0, lowerIsBetter: false },
        { label: "Carbs (g)", valA: a.carbs_per_100g || a.carbs || 0, valB: b.carbs_per_100g || b.carbs || 0, lowerIsBetter: true },
        { label: "Fat (g)", valA: a.fat_per_100g || a.fat || 0, valB: b.fat_per_100g || b.fat || 0, lowerIsBetter: true },
        { label: "Fiber (g)", valA: a.fiber_per_100g || (a.carbs_per_100g * 0.12) || 0, valB: b.fiber_per_100g || (b.carbs_per_100g * 0.12) || 0, lowerIsBetter: false },
        { label: "Sugar (g)", valA: a.sugar_per_100g || (a.carbs_per_100g * 0.08) || 0, valB: b.sugar_per_100g || (b.carbs_per_100g * 0.08) || 0, lowerIsBetter: true },
        { label: "Sodium (mg)", valA: a.sodium_per_100g || 250, valB: b.sodium_per_100g || 250, lowerIsBetter: true }
    ];

    const body = document.getElementById("results-compare-table-body");
    if (body) {
        body.innerHTML = metrics.map(m => {
            const scoreA = parseFloat(m.valA);
            const scoreB = parseFloat(m.valB);

            let classA = "";
            let classB = "";

            if (Math.abs(scoreA - scoreB) < 0.05) {
                classA = "text-warning"; // similar
                classB = "text-warning";
            } else {
                const isABetter = m.lowerIsBetter ? (scoreA < scoreB) : (scoreA > scoreB);
                classA = isABetter ? "text-success fw-bold" : "text-danger";
                classB = isABetter ? "text-danger" : "text-success fw-bold";
            }

            return `
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.03);">
                    <td class="text-secondary text-start py-2 small">${m.label}</td>
                    <td class="${classA} py-2">${scoreA.toFixed(1)}</td>
                    <td class="${classB} py-2">${scoreB.toFixed(1)}</td>
                </tr>
            `;
        }).join("");
    }

    // Calculate Healthier Choice winner
    const scoreA = clientComputeHealthScore(a, 100);
    const scoreB = clientComputeHealthScore(b, 100);

    const winnerCard = document.getElementById("healthier-choice-card");
    if (winnerCard) {
        winnerCard.classList.remove("d-none");
        const winnerName = document.getElementById("healthier-winner-name");
        const winnerReason = document.getElementById("healthier-winner-reason");
        const innerCard = document.getElementById("healthier-choice-inner");

        if (scoreA >= scoreB) {
            winnerName.textContent = a.display_name;
            winnerName.className = "text-success fw-bold mb-1";
            winnerReason.textContent = `${a.display_name} has a superior AI health rating (${scoreA}/100 vs ${scoreB}/100) due to better balanced nutritional values.`;
            if (innerCard) innerCard.className = "healthier-choice-card win-a";
        } else {
            winnerName.textContent = b.display_name;
            winnerName.className = "text-primary fw-bold mb-1";
            winnerReason.textContent = `${b.display_name} is the recommended healthier choice (${scoreB}/100 vs ${scoreA}/100) due to lower calories, fat, or sugar content.`;
            if (innerCard) innerCard.className = "healthier-choice-card win-b";
        }
    }

    // Refresh charts if in charts view
    if (activeComparisonView === 'charts') {
        renderComparisonCharts();
    }
}

function setComparisonView(view) {
    activeComparisonView = view;
    document.getElementById("tab-btn-table").classList.toggle("active", view === 'table');
    document.getElementById("tab-btn-charts").classList.toggle("active", view === 'charts');

    document.getElementById("compare-table-view").classList.toggle("d-none", view !== 'table');
    document.getElementById("compare-charts-view").classList.toggle("d-none", view !== 'charts');

    if (view === 'charts') {
        setTimeout(renderComparisonCharts, 50);
    }
}

function renderComparisonCharts() {
    if (!selectedCompareA || !selectedCompareB) return;

    const a = selectedCompareA;
    const b = selectedCompareB;

    const valA = {
        carbs: a.carbs_per_100g || a.carbs || 0,
        protein: a.protein_per_100g || a.protein || 0,
        fat: a.fat_per_100g || a.fat || 0
    };
    const valB = {
        carbs: b.carbs_per_100g || b.carbs || 0,
        protein: b.protein_per_100g || b.protein || 0,
        fat: b.fat_per_100g || b.fat || 0
    };

    if (compareBarChartInstance) compareBarChartInstance.destroy();
    if (compareRadarChartInstance) compareRadarChartInstance.destroy();

    // 1. Render Bar Chart
    const ctxBar = document.getElementById('compareBarChart').getContext('2d');
    compareBarChartInstance = new Chart(ctxBar, {
        type: 'bar',
        data: {
            labels: ['Carbs (g)', 'Protein (g)', 'Fat (g)'],
            datasets: [
                {
                    label: a.display_name,
                    data: [valA.carbs, valA.protein, valA.fat],
                    backgroundColor: 'rgba(59, 130, 246, 0.75)',
                    borderColor: '#3b82f6',
                    borderWidth: 1
                },
                {
                    label: b.display_name,
                    data: [valB.carbs, valB.protein, valB.fat],
                    backgroundColor: 'rgba(245, 158, 11, 0.75)',
                    borderColor: '#f59e0b',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    // 2. Render Radar Chart
    const getRadVal = (food) => {
        const carbs = food.carbs_per_100g || food.carbs || 0;
        return [
            Math.min(((food.calories_per_100g || food.calories || 0) / 400) * 100, 100),
            Math.min((carbs / 80) * 100, 100),
            Math.min(((food.protein_per_100g || food.protein || 0) / 30) * 100, 100),
            Math.min(((food.fat_per_100g || food.fat || 0) / 30) * 100, 100),
            Math.min(((food.fiber_per_100g || (carbs * 0.12)) / 10) * 100, 100),
            Math.min(((food.sugar_per_100g || (carbs * 0.08)) / 25) * 100, 100)
        ];
    };

    const ctxRadar = document.getElementById('compareRadarChart').getContext('2d');
    compareRadarChartInstance = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Calories', 'Carbs', 'Protein', 'Fat', 'Fiber', 'Sugar'],
            datasets: [
                {
                    label: a.display_name,
                    data: getRadVal(a),
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderColor: '#3b82f6',
                    pointBackgroundColor: '#3b82f6',
                    borderWidth: 1.5
                },
                {
                    label: b.display_name,
                    data: getRadVal(b),
                    backgroundColor: 'rgba(245, 158, 11, 0.15)',
                    borderColor: '#f59e0b',
                    pointBackgroundColor: '#f59e0b',
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8', font: { size: 10 } } }
            },
            scales: {
                r: {
                    angleLines: { color: 'rgba(255,255,255,0.05)' },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    pointLabels: { color: '#94a3b8', font: { size: 9 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

function renderTop5(predictions) {
    const container = document.getElementById("res-top-5-list");
    if (!container) return;

    container.innerHTML = predictions.map(p => {
        const conf = p.confidence || 0;
        let barColor = "bg-success";
        if (conf < 70) barColor = "bg-danger";
        else if (conf < 90) barColor = "bg-warning";

        return `
            <div class="top-5-row">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <span class="text-light small fw-medium">${p.rank}. ${p.food_name}</span>
                    <span class="text-secondary small fw-semibold">${conf.toFixed(2)}%</span>
                </div>
                <div class="progress" style="height: 6px;">
                    <div class="progress-bar ${barColor}" style="width: ${conf}%;"></div>
                </div>
            </div>
        `;
    }).join("");
}

/* ─────────────────────────────────────────────────────────────────────────────
   7. UI STATE MANAGEMENT (Show/Hide Cards)
   ───────────────────────────────────────────────────────────────────────────── */
function showCard(cardName) {
    const scannerSec = document.getElementById("scanner-section");
    const resultsSec = document.getElementById("results-section");

    if (cardName === "results") {
        if (scannerSec) scannerSec.classList.add("d-none");
        if (resultsSec) resultsSec.classList.remove("d-none");
    } else {
        if (resultsSec) resultsSec.classList.add("d-none");
        if (scannerSec) scannerSec.classList.remove("d-none");

        // Card IDs inside scanner right column
        const cards = {
            awaiting: "awaiting-scan-card",
            loading: "loading-scan-card",
            error: "error-scan-card"
        };

        Object.values(cards).forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add("d-none");
        });

        const target = cards[cardName];
        if (target) {
            const el = document.getElementById(target);
            if (el) el.classList.remove("d-none");
        }
    }
}

function resetToScanner() {
    resetPreview();
}

function showError(title, message) {
    const errTitle = document.getElementById("error-title");
    const errMsg = document.getElementById("error-message");
    if (errTitle) errTitle.innerText = title;
    if (errMsg) errMsg.innerText = message;
    showCard("error");
}

function resetPreview() {
    selectedFile = null;

    const previewPanel = document.getElementById("preview-panel");
    const previewImg = document.getElementById("preview-image-elem");
    const fileInput = document.getElementById("image-file-input");

    if (previewPanel) previewPanel.classList.add("d-none");
    if (previewImg) previewImg.src = "";
    if (fileInput) fileInput.value = "";

    showCard("awaiting");
}

function resetPreviewSilent() {
    selectedFile = null;
    const previewPanel = document.getElementById("preview-panel");
    const previewImg = document.getElementById("preview-image-elem");
    const fileInput = document.getElementById("image-file-input");

    if (previewPanel) previewPanel.classList.add("d-none");
    if (previewImg) previewImg.src = "";
    if (fileInput) fileInput.value = "";
}

/* ─────────────────────────────────────────────────────────────────────────────
   8. UTILITIES
   ───────────────────────────────────────────────────────────────────────────── */
function setTextById(id, text) {
    const el = document.getElementById(id);
    if (el) el.innerText = text;
}

/* ─────────────────────────────────────────────────────────────────────────────
   9. AI PLATE MUTATOR — Diet Health Swapper
   ───────────────────────────────────────────────────────────────────────────── */
function resetMutatorUI() {
    const placeholder = document.getElementById("mutator-placeholder");
    const results = document.getElementById("mutator-results");
    const loading = document.getElementById("mutator-loading");

    if (placeholder) placeholder.classList.remove("d-none");
    if (results) results.classList.add("d-none");
    if (loading) loading.classList.add("d-none");

    // Deactivate all pills
    document.querySelectorAll(".mutator-pill").forEach(p => p.classList.remove("active"));
}

function selectDiet(diet) {
    if (!lastPredictedFoodClass) return;

    // Toggle off if same diet clicked again
    if (currentMutatorDiet === diet) {
        currentMutatorDiet = null;
        resetMutatorUI();
        return;
    }

    currentMutatorDiet = diet;

    // Activate the correct pill
    document.querySelectorAll(".mutator-pill").forEach(p => {
        p.classList.toggle("active", p.dataset.diet === diet);
    });

    // Show loading
    const placeholder = document.getElementById("mutator-placeholder");
    const results = document.getElementById("mutator-results");
    const loading = document.getElementById("mutator-loading");

    if (placeholder) placeholder.classList.add("d-none");
    if (results) results.classList.add("d-none");
    if (loading) loading.classList.remove("d-none");

    // Call the mutate API
    fetch("/api/mutate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            food_class: lastPredictedFoodClass,
            diet: diet,
            weight: lastPredictedWeight
        })
    })
        .then(res => res.json())
        .then(data => {
            if (loading) loading.classList.add("d-none");

            if (!data.success) {
                if (placeholder) {
                    placeholder.innerHTML = `<i class="fa-solid fa-circle-exclamation text-danger fs-2 mb-2 d-block"></i><p class="text-danger small mb-0">${data.error || 'Mutation failed.'}</p>`;
                    placeholder.classList.remove("d-none");
                }
                return;
            }

            renderMutatorResults(data);
        })
        .catch(err => {
            console.error("Mutator API error:", err);
            if (loading) loading.classList.add("d-none");
            if (placeholder) {
                placeholder.innerHTML = `<i class="fa-solid fa-circle-exclamation text-danger fs-2 mb-2 d-block"></i><p class="text-danger small mb-0">Could not reach the server.</p>`;
                placeholder.classList.remove("d-none");
            }
        });
}

function renderMutatorResults(data) {
    const results = document.getElementById("mutator-results");
    if (!results) return;
    results.classList.remove("d-none");

    // Diet badge
    const badge = document.getElementById("mut-diet-badge");
    if (badge) badge.style.backgroundColor = data.diet_color + "22";
    if (badge) badge.style.color = data.diet_color;
    if (badge) badge.style.border = `1px solid ${data.diet_color}44`;

    const icon = document.getElementById("mut-diet-icon");
    if (icon) icon.className = `fa-solid ${data.diet_icon} me-1`;

    setTextById("mut-diet-label", data.diet_label);
    setTextById("mut-food-name", data.display_name);

    // Original values
    setTextById("mut-orig-cal", data.original.calories.toFixed(1) + " kcal");
    setTextById("mut-orig-carbs", data.original.carbs.toFixed(1) + "g");
    setTextById("mut-orig-protein", data.original.protein.toFixed(1) + "g");
    setTextById("mut-orig-fat", data.original.fat.toFixed(1) + "g");

    // Mutated values
    setTextById("mut-new-cal", data.mutated.calories.toFixed(1) + " kcal");
    setTextById("mut-new-carbs", data.mutated.carbs.toFixed(1) + "g");
    setTextById("mut-new-protein", data.mutated.protein.toFixed(1) + "g");
    setTextById("mut-new-fat", data.mutated.fat.toFixed(1) + "g");

    // Mutated column label color
    const mutLabel = document.getElementById("mut-mutated-label");
    if (mutLabel) mutLabel.style.color = data.diet_color;

    // Delta badges with animation
    renderDelta("mut-delta-cal", data.delta.calories, "kcal");
    renderDelta("mut-delta-carbs", data.delta.carbs, "g");
    renderDelta("mut-delta-protein", data.delta.protein, "g");
    renderDelta("mut-delta-fat", data.delta.fat, "g");

    // Ingredient swaps
    const swapsSection = document.getElementById("mut-swaps-section");
    const swapsList = document.getElementById("mut-swaps-list");
    if (data.swaps && data.swaps.length > 0 && swapsSection && swapsList) {
        swapsSection.classList.remove("d-none");
        swapsList.innerHTML = data.swaps.map((s, i) => `
            <div class="mutator-swap-row" style="animation-delay: ${i * 0.08}s">
                <div class="d-flex align-items-center gap-2 flex-grow-1">
                    <span class="mutator-swap-original">${s.original}</span>
                    <i class="fa-solid fa-arrow-right-long text-info"></i>
                    <span class="mutator-swap-replacement">${s.replacement}</span>
                </div>
                <span class="mutator-swap-impact">${s.impact}</span>
            </div>
        `).join("");
    } else if (swapsSection) {
        swapsSection.classList.add("d-none");
    }

    // Recipe tip
    setTextById("mut-recipe-tip", data.recipe_tip);

    // Animate the results container in
    results.style.animation = "none";
    results.offsetHeight; // trigger reflow
    results.style.animation = "mutatorFadeIn 0.4s ease-out forwards";
}

function renderDelta(elementId, value, unit) {
    const el = document.getElementById(elementId);
    if (!el) return;

    if (Math.abs(value) < 0.1) {
        el.textContent = "—";
        el.className = "mutator-delta neutral";
        return;
    }

    const isPositive = value > 0;
    const sign = isPositive ? "+" : "";
    el.textContent = sign + value.toFixed(1) + unit;

    // For calories, carbs, fat: negative = green (good), positive = red (bad)
    // For protein: positive = green (good), negative = red (bad)
    const isProtein = elementId.includes("protein");
    let isGood;
    if (isProtein) {
        isGood = isPositive; // More protein = good
    } else {
        isGood = !isPositive; // Less cal/carbs/fat = good
    }

    el.className = `mutator-delta ${isGood ? 'good' : 'bad'}`;

    // Trigger pop animation
    el.style.animation = "none";
    el.offsetHeight;
    el.style.animation = "deltaPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards";
}

/* ─────────────────────────────────────────────────────────────────────────────
   10. UPGRADED FEATURE INTEGRATIONS (INSIGHTS, TRACKERS, SECOND OPINION, BMI, COMPARE)
   ───────────────────────────────────────────────────────────────────────────── */

function populateInsights(data) {
    const panel = document.getElementById("ai-insights-panel");
    const freshnessBadge = document.getElementById("freshness-badge");
    const freshnessDesc = document.getElementById("freshness-desc");
    const altText = document.getElementById("healthier-alt-text");
    const recipesList = document.getElementById("recipe-suggestions-list");

    if (!panel) return;

    // Show panel
    panel.classList.remove("d-none");

    // Freshness
    if (data.freshness && freshnessBadge && freshnessDesc) {
        freshnessBadge.textContent = data.freshness.status;
        freshnessBadge.className = `badge fs-6 px-3 py-2 ${data.freshness.css_class}`;
        freshnessDesc.textContent = data.freshness.desc;
    }

    // Healthier Alternative
    if (altText) {
        altText.innerHTML = `Instead of <strong>${data.food_name}</strong>, try: <span class="text-success fw-bold">${data.healthier_alternative}</span>`;
    }

    // Recipes Suggestions
    if (recipesList && data.recipes) {
        recipesList.innerHTML = data.recipes.map(r => `
            <div class="p-3 bg-dark bg-opacity-25 rounded border border-secondary-subtle">
                <div class="d-flex justify-content-between align-items-center mb-1">
                    <strong class="text-light small">${r.name}</strong>
                    <span class="badge bg-success-subtle text-success small">${r.calories}</span>
                </div>
                <div class="text-secondary small mb-1"><strong>Ingredients:</strong> ${r.ingredients}</div>
            </div>
        `).join("");
    }
}

function populateSecondOpinion(data) {
    const panel = document.getElementById("second-opinion-panel");
    const container = document.getElementById("second-opinion-choices");

    if (!panel || !container) return;

    if (data.requires_confirmation && data.top_5 && data.top_5.length > 1) {
        panel.classList.remove("d-none");
        container.innerHTML = data.top_5.slice(0, 3).map(p => `
            <button type="button" class="btn btn-outline-info text-start w-100 py-2" onclick="confirmSecondOpinion('${p.class_key}', ${data.nutrition.weight})">
                <i class="fa-solid fa-circle-check me-2"></i>Yes, it is <strong>${p.food_name}</strong> (${p.confidence.toFixed(0)}%)
            </button>
        `).join("") + `
            <button type="button" class="btn btn-outline-secondary text-start w-100 py-2" onclick="dismissSecondOpinion()">
                <i class="fa-solid fa-xmark me-2"></i>None of these
            </button>
        `;
    } else {
        panel.classList.add("d-none");
    }
}

function dismissSecondOpinion() {
    const panel = document.getElementById("second-opinion-panel");
    if (panel) panel.classList.add("d-none");
}

function confirmSecondOpinion(classKey, weight) {
    if (!lastEntryId) return;

    fetch("/api/predict/confirm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ entry_id: lastEntryId, class_key: classKey, weight: weight })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                // Update UI elements with confirmed nutrition
                const n = data.updated_nutrition;
                lastPredictedFoodClass = data.food_name;

                setTextById("res-food-name", data.food_name);
                setTextById("res-kcal", n.calories.toFixed(1) + " kcal");
                setTextById("res-health", n.health_rating);
                setTextById("res-carbs", n.carbs.toFixed(1) + "g");
                setTextById("res-protein", n.protein.toFixed(1) + "g");
                setTextById("res-fat", n.fat.toFixed(1) + "g");
                setTextById("res-serving", n.serving_size);
                setTextById("res-fiber", n.fiber.toFixed(1) + "g");
                setTextById("res-sugar", n.sugar.toFixed(1) + "g");
                setTextById("res-sodium", Math.round(n.sodium) + "mg");

                // Hide confirmation panel
                dismissSecondOpinion();

                // Refresh trackers & insights
                data.recipes && populateInsights({
                    food_name: data.food_name,
                    healthier_alternative: data.healthier_alternative,
                    recipes: data.recipes
                });
                refreshDailyTracker();
            }
        })
        .catch(err => console.error("Confirm prediction error:", err));
}

function refreshDailyTracker() {
    const panel = document.getElementById("daily-tracker-panel");
    const ring = document.getElementById("calorie-ring-progress");
    const consumedVal = document.getElementById("tracker-cal-consumed");
    const goalVal = document.getElementById("tracker-cal-goal");
    const carbsVal = document.getElementById("tracker-carbs");
    const proteinVal = document.getElementById("tracker-protein");
    const fatVal = document.getElementById("tracker-fat");
    const remainingVal = document.getElementById("tracker-remaining");
    const warningsContainer = document.getElementById("diet-warnings-container");

    if (!panel) return;

    fetch("/api/history/today")
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                panel.classList.remove("d-none");

                const t = data.totals;
                const goal = t.calorie_goal;
                const consumed = t.calories;

                if (consumedVal) consumedVal.textContent = consumed.toFixed(0);
                if (goalVal) goalVal.textContent = goal.toFixed(0);
                if (carbsVal) carbsVal.textContent = t.carbs.toFixed(1) + "g";
                if (proteinVal) proteinVal.textContent = t.protein.toFixed(1) + "g";
                if (fatVal) fatVal.textContent = t.fat.toFixed(1) + "g";

                // Remaining Calorie Logic
                if (remainingVal) {
                    if (t.remaining >= 0) {
                        remainingVal.textContent = t.remaining.toFixed(0) + " kcal";
                        remainingVal.className = "text-success";
                    } else {
                        remainingVal.textContent = "+" + Math.abs(t.remaining).toFixed(0) + " kcal";
                        remainingVal.className = "text-danger";
                    }
                }

                // Circular ring progress update
                if (ring) {
                    const circumference = 326.7;
                    const offset = circumference * (1 - Math.min(consumed / goal, 1));
                    ring.style.strokeDashoffset = offset;
                }

                // Render diet warnings
                if (warningsContainer) {
                    if (t.warnings && t.warnings.length > 0) {
                        warningsContainer.innerHTML = t.warnings.map(w => `
                        <div class="alert alert-${w.level} d-flex align-items-center gap-2 mb-2 p-2 border-0" role="alert">
                            <i class="fa-solid ${w.icon}"></i>
                            <span class="small">${w.msg}</span>
                        </div>
                    `).join("");
                    } else {
                        warningsContainer.innerHTML = `
                        <div class="alert alert-success d-flex align-items-center gap-2 mb-0 p-2 border-0" role="alert">
                            <i class="fa-solid fa-circle-check"></i>
                            <span class="small">All metrics within recommended targets!</span>
                        </div>
                    `;
                    }
                }
            }
        })
        .catch(err => console.error("Fetch today totals error:", err));
}

// Initialise daily tracker update on load
document.addEventListener("DOMContentLoaded", () => {
    refreshDailyTracker();
});

function openBMIModal() {
    const modal = new bootstrap.Modal(document.getElementById("bmiModal"));
    modal.show();
}

function calculateBMI() {
    const age = document.getElementById("bmi-age").value;
    const gender = document.getElementById("bmi-gender").value;
    const weight = document.getElementById("bmi-weight").value;
    const height = document.getElementById("bmi-height").value;
    const activity = document.getElementById("bmi-activity").value;

    fetch("/api/bmi/calculate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ age, gender, weight, height, activity })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                const resBox = document.getElementById("bmi-result-box");
                resBox.classList.remove("d-none");

                setTextById("bmi-result-val", data.bmi);
                setTextById("bmi-result-cat", data.bmi_category);
                setTextById("bmi-result-cal", data.daily_calories.toFixed(0));

                // Instantly apply settings
                fetch("/api/settings", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ calorie_goal: data.daily_calories })
                })
                    .then(() => {
                        refreshDailyTracker();
                    });
            }
        })
        .catch(err => console.error("BMI calculate error:", err));
}

function runComparison() {
    const foodA = document.getElementById("compare-food-a").value;
    const foodB = document.getElementById("compare-food-b").value;
    const placeholder = document.getElementById("compare-placeholder");
    const container = document.getElementById("compare-results-container");
    const tableBody = document.getElementById("compare-table-body");
    const headerA = document.getElementById("compare-header-a");
    const headerB = document.getElementById("compare-header-b");

    if (!foodA || !foodB) return;

    fetch("/api/nutrition/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ food_a: foodA, food_b: foodB })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                placeholder.classList.add("d-none");
                container.classList.remove("d-none");

                const a = data.food_a;
                const b = data.food_b;
                const w = data.winners;

                headerA.textContent = a.name;
                headerB.textContent = b.name;

                const rows = [
                    { label: "Calories", valA: a.calories.toFixed(0) + " kcal", valB: b.calories.toFixed(0) + " kcal", winner: w.calories },
                    { label: "Carbs", valA: a.carbs.toFixed(1) + "g", valB: b.carbs.toFixed(1) + "g", winner: w.carbs },
                    { label: "Protein", valA: a.protein.toFixed(1) + "g", valB: b.protein.toFixed(1) + "g", winner: w.protein },
                    { label: "Fat", valA: a.fat.toFixed(1) + "g", valB: b.fat.toFixed(1) + "g", winner: w.fat },
                    { label: "Fiber", valA: a.fiber.toFixed(1) + "g", valB: b.fiber.toFixed(1) + "g", winner: w.fiber },
                    { label: "Sugars", valA: a.sugar.toFixed(1) + "g", valB: b.sugar.toFixed(1) + "g", winner: w.sugar },
                    { label: "Sodium", valA: Math.round(a.sodium) + "mg", valB: Math.round(b.sodium) + "mg", winner: w.sodium }
                ];

                tableBody.innerHTML = rows.map(r => {
                    const cellA = r.winner === "A" ? `<td class="table-success fw-bold text-success">${r.valA}</td>` : `<td>${r.valA}</td>`;
                    const cellB = r.winner === "B" ? `<td class="table-success fw-bold text-success">${r.valB}</td>` : `<td>${r.valB}</td>`;
                    const winnerLabel = r.winner === "A" ? a.name : b.name;
                    return `
                    <tr>
                        <td class="text-secondary small fw-semibold">${r.label}</td>
                        ${cellA}
                        ${cellB}
                        <td class="text-success fw-bold"><i class="fa-solid fa-circle-check me-1"></i>${winnerLabel}</td>
                    </tr>
                `;
                }).join("");
            } else {
                alert(data.error || "Comparison error");
            }
        })
        .catch(err => console.error("Food comparison error:", err));
}

/* ─────────────────────────────────────────────────────────────────────────────
   COMMUNITY SHARING
   ───────────────────────────────────────────────────────────────────────────── */
function openSharePostModal() {
    if (!currentScanData) {
        alert("No scan data available to share. Please run a prediction first.");
        return;
    }
    const n = currentScanData.nutrition || {};
    document.getElementById("share-food-name").value = n.display_name || currentScanData.food_name || "Unknown";
    document.getElementById("share-calories").value = n.calories || 0;
    document.getElementById("share-protein").value = n.protein || 0;
    document.getElementById("share-carbs").value = n.carbs || 0;
    document.getElementById("share-fat").value = n.fat || 0;
    document.getElementById("share-health-rating").value = n.health_rating || "B";
    document.getElementById("share-image-path").value = currentScanData.image_url || "";

    // Autofill username from localStorage
    const savedUser = localStorage.getItem("community_username") || "";
    document.getElementById("share-username").value = savedUser;

    // Clear recipe fields
    document.getElementById("share-recipe-title").value = "";
    document.getElementById("share-recipe-instructions").value = "";

    const modal = new bootstrap.Modal(document.getElementById('sharePostModal'));
    modal.show();
}

// Bind form submission for sharing community post
document.addEventListener("DOMContentLoaded", () => {
    const shareForm = document.getElementById("share-post-form");
    if (shareForm) {
        shareForm.addEventListener("submit", function (e) {
            e.preventDefault();
            const username = document.getElementById("share-username").value.trim();
            if (!username) {
                alert("Username is required.");
                return;
            }
            // Save username to localStorage for next time
            localStorage.setItem("community_username", username);

            const payload = {
                username: username,
                food_name: document.getElementById("share-food-name").value,
                calories: parseFloat(document.getElementById("share-calories").value),
                protein: parseFloat(document.getElementById("share-protein").value),
                carbs: parseFloat(document.getElementById("share-carbs").value),
                fat: parseFloat(document.getElementById("share-fat").value),
                health_rating: document.getElementById("share-health-rating").value,
                image_path: document.getElementById("share-image-path").value,
                recipe_title: document.getElementById("share-recipe-title").value.trim(),
                recipe_instructions: document.getElementById("share-recipe-instructions").value.trim()
            };

            fetch('/api/community/posts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        const modalEl = document.getElementById('sharePostModal');
                        const modalInstance = bootstrap.Modal.getInstance(modalEl);
                        if (modalInstance) modalInstance.hide();
                        window.location.href = "/community";
                    } else {
                        alert(data.error || "Failed to share post.");
                    }
                })
                .catch(err => {
                    console.error("Share error:", err);
                    alert("A network error occurred while sharing.");
                });
        });
    }
});

/* ─────────────────────────────────────────────────────────────────────────────
   FEATURE 4: VOICE READ-OUT OF RESULTS (Web Speech API)
   ───────────────────────────────────────────────────────────────────────────── */
function speakPredictionSummary() {
    if (!('speechSynthesis' in window)) {
        alert("Web Speech API is not supported in your browser.");
        return;
    }

    // Stop any ongoing speech
    window.speechSynthesis.cancel();

    if (!currentScanData) {
        alert("No prediction data to read.");
        return;
    }

    const n = currentScanData.nutrition || {};
    const foodName = n.display_name || currentScanData.food_name || "Food item";
    const calories = Math.round(n.calories || 0);
    const giCat = n.gi_category || "Medium";
    const giScore = n.glycemic_index || 55;
    const rating = n.health_rating || "B";
    const allergens = (n.allergens && n.allergens.length > 0) ? n.allergens.join(" and ") : "no common allergens";

    const speechText = `${foodName} detected. Estimated portion contains ${calories} calories. ` +
        `Glycemic Index is ${giCat} with a score of ${giScore}. ` +
        `Health rating is ${rating}. ` +
        `Contains ${allergens}.`;

    const utterance = new SpeechSynthesisUtterance(speechText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';

    window.speechSynthesis.speak(utterance);
}

/* ─────────────────────────────────────────────────────────────────────────────
   FEATURE 9: PRINTABLE NUTRITION LABEL (PDF EXPORT)
   ───────────────────────────────────────────────────────────────────────────── */
function downloadPDFReport(entryId) {
    const id = entryId || lastEntryId || (currentScanData ? currentScanData.entry_id : null);
    if (!id) {
        alert("No scan entry available to generate PDF report. Please run a prediction first.");
        return;
    }
    window.open(`/api/report/pdf/${id}`, '_blank');
}

/* ─────────────────────────────────────────────────────────────────────────────
   FEATURE 12: WATER INTAKE TRACKER (WEB JS)
   ───────────────────────────────────────────────────────────────────────────── */
function fetchWaterIntake() {
    fetch('/api/water/today')
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                updateWaterUI(data);
            }
        })
        .catch(err => console.log("Water fetch error:", err));
}

function updateWaterIntake(delta) {
    fetch('/api/water/log', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta: delta })
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                updateWaterUI(data);
            }
        })
        .catch(err => console.log("Water log error:", err));
}

function updateWaterUI(data) {
    const glasses = data.glasses || 0;
    const target = data.target_glasses || 8;
    const volume = data.volume_ml || (glasses * 250);
    const pct = data.percentage || Math.min(100, Math.round((glasses / target) * 100));

    setTextById("water-badge", `${glasses} / ${target} Glasses`);
    setTextById("water-glasses-count", `${glasses} / ${target}`);
    setTextById("water-volume-ml", volume);
    setTextById("water-percentage", `${pct}%`);

    const bar = document.getElementById("water-progress-bar");
    if (bar) {
        bar.style.width = pct + "%";
    }
}

// Auto-load water intake on page load
document.addEventListener("DOMContentLoaded", () => {
    fetchWaterIntake();
});

/* ─────────────────────────────────────────────────────────────────────────────
   FEATURE 11: PACKAGED FOOD BARCODE SEARCH (WEB JS)
   ───────────────────────────────────────────────────────────────────────────── */
function searchBarcodeHome() {
    const input = document.getElementById("barcode-quick-input");
    if (!input || !input.value.trim()) {
        alert("Please enter a barcode number.");
        return;
    }
    const code = input.value.trim();
    fetch(`/api/barcode/${code}`)
        .then(res => res.json())
        .then(data => {
            if (data.success && data.found) {
                const p = data.product;
                alert(`📦 ${p.brand} - ${p.product_name}\n\n` +
                    `Calories: ${p.calories_per_100g} kcal / 100g\n` +
                    `Protein: ${p.protein_per_100g}g | Carbs: ${p.carbs_per_100g}g | Fat: ${p.fat_per_100g}g\n` +
                    `Sodium: ${p.sodium_per_100g}mg\n` +
                    `Health Grade: ${p.health_rating}\n` +
                    `Allergens: ${(p.allergens || []).join(', ') || 'None'}`);
            } else {
                alert(`Barcode '${code}' not found in database. Try sample codes: 8901058852312 or 8901491101820.`);
            }
        })
        .catch(err => alert("Error looking up barcode."));
}

