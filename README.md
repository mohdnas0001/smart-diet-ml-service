# Smart Diet ML Service 🧠🍽️

AI-powered **Food Recognition & Nutrient Estimation** microservice for the Smart Diet and Food Analyzer system.

---

## What is this?

This is a **Python/FastAPI** microservice that:
1. Accepts a food image via HTTP
2. Detects food items using **YOLOv8**
3. Classifies them using **EfficientNet-B4**
4. Estimates portion sizes using a **Bayesian prior model**
5. Returns detailed **65+ nutrient profiles** per food item

It has a built-in **DEMO MODE** — it works immediately without any `.pt` model files by returning realistic simulated detections.

---

## System Integration

```
📱 React Native App (Member 2)
        ↓ HTTP
🖥️ ASP.NET Core C# Backend (Member 3)
        ↓ Internal HTTP POST /api/predict
🧠 Python ML Service  ← THIS REPO
        ↓ Async HTTP fallback
🌐 USDA API / Nutritionix API
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| ML models | YOLOv8 (detection) + EfficientNet-B4 (classification) |
| Data validation | Pydantic v2 |
| External APIs | USDA FoodData Central, Nutritionix |
| Food database | Nigerian Foods DB (150+ foods, 65+ nutrients each) |
| Containerisation | Docker + docker-compose |
| Testing | pytest + pytest-asyncio |
| Training | PyTorch, timm, ultralytics, albumentations |

---

## Folder Structure

```
smart-diet-ml-service/
├── app/                        ← Main application code
│   ├── main.py                 ← FastAPI app, CORS, lifespan startup
│   ├── config.py               ← Settings (pydantic-settings, reads .env)
│   ├── routes/
│   │   ├── predict.py          ← POST /api/predict
│   │   ├── health.py           ← GET  /api/health
│   │   └── nutrients.py        ← GET  /api/nutrients/{food_name}
│   ├── schemas/
│   │   ├── response.py         ← NutrientProfile (65+ fields), FoodItem, AnalysisResponse
│   │   ├── request.py          ← AnalysisRequest
│   │   └── common.py           ← FoodRegion, MealType enums
│   ├── models/
│   │   ├── detector.py         ← FoodDetector (YOLOv8 wrapper + demo mode)
│   │   ├── classifier.py       ← FoodClassifier (EfficientNet-B4 + demo mode)
│   │   └── portion_estimator.py← PortionEstimator (Bayesian priors)
│   ├── services/
│   │   ├── analysis_pipeline.py← Orchestrates full 5-stage pipeline
│   │   ├── preprocessing.py    ← Image resize, normalise, EXIF rotation
│   │   ├── nutrient_service.py ← Hierarchical DB lookup (Nigerian → USDA → Nutritionix)
│   │   ├── food_mapper.py      ← Fuzzy string matching (thefuzz)
│   │   ├── usda_client.py      ← Async USDA FoodData Central client
│   │   └── nutritionix_client.py← Async Nutritionix client
│   └── utils/
│       ├── image_utils.py      ← load, strip EXIF, validate image
│       ├── atwater.py          ← Calorie validation (4/4/9/2 factors)
│       └── logger.py           ← Structured logging
├── data/                       ← JSON databases (no ML weights required)
│   ├── nigerian_foods.json     ← 150+ Nigerian foods, 65+ nutrients each
│   ├── food_categories.json    ← 293 food categories (Nigerian + international)
│   ├── portion_priors.json     ← Bayesian priors for portion estimation
│   ├── food_density_table.json ← Food density in g/cm³
│   └── rda_values.json         ← Recommended daily allowances for all nutrients
├── training/                   ← ML training scripts (requires GPU + dataset)
│   ├── train_classifier.py     ← EfficientNet-B4 training
│   ├── train_detector.py       ← YOLOv8 training
│   ├── train_portion_estimator.py← Fits portion priors from annotations
│   ├── evaluate.py             ← Top-1 / Top-5 accuracy evaluation
│   ├── export_model.py         ← Export to ONNX / TFLite
│   ├── augmentation.py         ← Train/val transforms
│   └── dataset.py              ← FoodDataset (ImageFolder wrapper)
├── tests/                      ← Pytest test suite (35 tests, all pass)
│   ├── conftest.py             ← Fixtures: test_client, sample_image_bytes
│   ├── test_health_endpoint.py
│   ├── test_predict_endpoint.py
│   ├── test_nutrient_endpoint.py
│   ├── test_nutrient_service.py
│   ├── test_preprocessing.py
│   ├── test_food_mapper.py
│   ├── test_atwater.py
│   └── test_analysis_pipeline.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt            ← Runtime dependencies
├── requirements-dev.txt        ← + pytest for testing
├── requirements-training.txt   ← + torch, ultralytics, timm, etc.
├── .env.example                ← Copy to .env and fill in your keys
├── pyproject.toml
└── pytest.ini
```

---

## ML Techniques & Concepts Involved

### 1. Object Detection — YOLOv8
- **What it does**: Finds bounding boxes around food items in the image
- **Architecture**: YOLO (You Only Look Once) — a single-pass CNN that divides the image into a grid and predicts boxes + classes simultaneously
- **Key concepts**: Anchor boxes, Non-Maximum Suppression (NMS), IoU loss

### 2. Image Classification — EfficientNet-B4
- **What it does**: Given a detected food crop, classifies it into one of 274 categories
- **Architecture**: EfficientNet scales width, depth, and resolution together using a compound coefficient. B4 is the 4th size variant
- **Key concepts**: Transfer learning (pretrained on ImageNet), compound scaling, depthwise separable convolutions

### 3. Portion Estimation — Bayesian Priors
- **What it does**: Estimates the weight in grams of the detected food
- **Technique**: Uses a Gaussian prior `N(mean, std)` per food type, combined with the bounding-box area (larger box → more food)
- **Key concepts**: Bayesian inference, prior distributions

### 4. Nutrient Estimation — Hierarchical Lookup
- **What it does**: Returns 65+ nutrient values (calories, protein, carbs, vitamins, minerals, amino acids, etc.)
- **Technique**: Looks up per-100g values in the Nigerian Food DB, then scales to the estimated portion
- **Key concepts**: Atwater energy factors (4 kcal/g protein, 4 kcal/g carbs, 9 kcal/g fat, 2 kcal/g fiber)

### 5. Fuzzy String Matching
- **Library**: `thefuzz` (Levenshtein distance)
- **Use case**: Maps arbitrary model output labels to canonical food names in the database

### 6. Data Augmentation (training)
- Random crop, horizontal flip, colour jitter, rotation
- Normalisation using ImageNet mean/std

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/api/health` | Health check, demo mode status |
| `POST` | `/api/predict` | Upload image → get food items + nutrients |
| `GET` | `/api/nutrients/{food_name}` | Lookup nutrients (optional `?portion_grams=200`) |
| `GET` | `/docs` | Interactive Swagger UI |

### Example: POST /api/predict

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@my_food_photo.jpg"
```

Response:
```json
{
  "analysis_id": "uuid-here",
  "image_width": 640,
  "image_height": 480,
  "food_items": [
    {
      "name": "jollof_rice",
      "confidence": 0.92,
      "bounding_box": {"x": 0.05, "y": 0.12, "width": 0.38, "height": 0.45},
      "portion_grams": 320.5,
      "food_region": "nigerian",
      "nutrients": {
        "calories": 538.4, "carbohydrates": 87.4, "protein": 12.2, "total_fat": 16.6, ...
      }
    }
  ],
  "total_calories": 538.4,
  "processing_time_ms": 45.2,
  "warnings": ["Running in DEMO mode — model weights not loaded"]
}
```

---

## How to Run Locally

### 1. Clone and setup
```bash
git clone https://github.com/mohdnas0001/smart-diet-ml-service
cd smart-diet-ml-service
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — at minimum set DEMO_MODE=true (default)
```

### 3. Run the service
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/docs** for the interactive API explorer.

### 4. Run tests
```bash
pytest tests/ -v
```

All **35 tests** should pass without any model files (demo mode).

---

## How to Run with Docker

```bash
cp .env.example .env
docker compose -f docker/docker-compose.yml up --build
```

---

## How to Train (requires GPU + dataset)

```bash
pip install -r requirements-training.txt

# Train food classifier (EfficientNet-B4)
python training/train_classifier.py --data_dir ./datasets/food --epochs 50

# Train food detector (YOLOv8)
python training/train_detector.py --data ./datasets/food.yaml --epochs 100

# Evaluate
python training/evaluate.py --model_path ./models/classifier.pt --data_dir ./datasets/food/test

# Export to ONNX
python training/export_model.py --model_path ./models/classifier.pt --format onnx
```

Place the exported `.pt` files in `./models/` and set `DEMO_MODE=false` in `.env`.

---

## Nigerian Food Database

The `data/nigerian_foods.json` contains **150+ authentic Nigerian foods** with **65+ nutrients per food** (per 100g):

- Macros: calories, carbs, protein, fat (total/saturated/mono/poly/trans)
- Minerals: calcium, iron, magnesium, phosphorus, zinc, copper, selenium, iodine, and more
- Vitamins: A, C, D, E, K, B1-B12, folate, biotin, choline
- Fatty acids: omega-3, omega-6, EPA, DHA, ALA
- Amino acids: all 9 essential amino acids (tryptophan, leucine, lysine, etc.)

Examples: jollof rice, egusi soup, suya, akara, pounded yam, moin moin, chin chin, zobo, kilishi, fufu, amala, puff puff, and many more.

---

## Module
**Member 1**: Food Recognition & Nutrient Estimation Module
