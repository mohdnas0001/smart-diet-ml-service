# Smart Diet ML Service Integration Guide

## Architecture Overview

This ML service is a **stateless microservice** that handles food detection, classification, and nutrient estimation. It integrates with the NestJS backend as follows:

```
Mobile App
    ↓
Backend (NestJS) ← HTTP calls
    ├─ JWT Auth
    ├─ User Management
    ├─ PostgreSQL (User data, History)
    └─ Forwards image to ML Service
         ↓
ML Service (FastAPI)
    ├─ Food Detection
    ├─ Classification
    ├─ Nutrient Calculation
    └─ Returns predictions
```

**Key principle:** ML Service does NOT interact with PostgreSQL. Backend handles all persistence.

---

## Environment Setup

### Backend Configuration

In your NestJS backend's `.env`:

```env
# ML Service Integration
ML_SERVICE_URL=http://localhost:8000           # or http://ml-service:8000 in Docker
ML_ANALYSIS_ENDPOINT=/api/predict
ML_ANALYSIS_FILE_FIELD=file
ML_ANALYSIS_TIMEOUT_MS=30000
```

### ML Service Configuration

In this ML Service's `.env`:

```env
PORT=8000
HOST=0.0.0.0
DEBUG=true
MODEL_DIR=./models
DATA_DIR=./data
USDA_API_KEY=your_key_here
NUTRITIONIX_APP_ID=your_id_here
NUTRITIONIX_APP_KEY=your_key_here
DEMO_MODE=false
APP_VERSION=1.0.0
```

---

## Local Development Setup

### 1. Start ML Service (Terminal 1)

```bash
cd /Users/mac/nile-projects/smart-diet-ml-service
cp .env.example .env
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

ML Service available at: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

### 2. Start Backend (Terminal 2)

```bash
cd /Users/mac/nile-projects/smart-diet-api-service
npm run start:dev
```

Backend available at: `http://localhost:3000`
- Swagger: `http://localhost:3000/docs`

### 3. Test the Integration

```bash
# 1. Register a user
curl -X POST http://localhost:3000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!","name":"Test User"}'

# 2. Upload an image (with auth token from register)
curl -X POST http://localhost:3000/analysis/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@path/to/food_image.jpg"

# 3. Get analysis history
curl -X GET http://localhost:3000/analysis/history \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Docker Compose Setup

### Development Setup (with PostgreSQL + pgAdmin)

```bash
cd /Users/mac/nile-projects/smart-diet-ml-service
docker compose up -d
```

This runs:
- ML Service: `http://localhost:8000`
- PostgreSQL: `localhost:5432` (backend will connect)
- Backend runs separately on port `3000`

### Production-Ready Setup

See `docker-compose.prod.yml` for a full integration setup with both services.

---

## API Response Format

The ML Service returns predictions in this format:

```json
{
  "analysis_id": "uuid-string",
  "image_width": 1024,
  "image_height": 768,
  "food_items": [
    {
      "name": "jollof_rice",
      "confidence": 0.92,
      "bounding_box": {
        "x": 100,
        "y": 150,
        "width": 300,
        "height": 250
      },
      "portion_grams": 320.5,
      "nutrients": {
        "calories": 538.4,
        "carbohydrates": 87.4,
        "protein": 12.2,
        "total_fat": 16.6,
        ...
      },
      "food_region": "nigerian"
    }
  ],
  "total_calories": 538.4,
  "total_macronutrients": {
    "total_calories": 538.4,
    "total_protein": 12.2,
    "total_carbs": 87.4,
    "total_fat": 16.6,
    "total_fiber": 2.1
  },
  "processing_time_ms": 1250,
  "model_versions": {
    "detector": "yolov8m",
    "classifier": "v1.0",
    "portion_estimator": "v1.0"
  },
  "warnings": []
}
```

Backend normalizes and stores this in PostgreSQL.

---

## Deployment Options

### Option 1: Separate Services (Recommended)
- ML Service: Container/VM on port `8000`
- Backend: Container/VM on port `3000`
- PostgreSQL: Managed database or container

Update backend `.env`:
```env
ML_SERVICE_URL=https://ml-service.yourdomain.com
```

### Option 2: Docker Swarm / Kubernetes
Both services in same cluster, communicate via internal DNS:
```env
ML_SERVICE_URL=http://ml-service:8000
```

### Option 3: Serverless (AWS Lambda / Google Cloud Run)
- Deploy backend as serverless function
- Keep ML service as containerized microservice
- Use managed PostgreSQL

---

## Health Checks

### ML Service Health
```bash
curl http://localhost:8000/api/health
# Response: {"status": "ok"}
```

### Backend Health
```bash
curl http://localhost:3000/health
# Response: {"status": "ok"}
```

---

## Performance Notes

- ML inference: **~1-2 seconds** per image (CPU)
- Backend request timeout: **30 seconds** (configurable)
- Recommend GPU for production (~200-400ms inference)
- Models cached in memory after first load

---

## Troubleshooting

### ML Service not responding
```bash
docker logs smart-diet-ml-service
```

### Backend can't reach ML Service
- Check `ML_SERVICE_URL` in backend `.env`
- Ensure ML service is running: `curl http://localhost:8000/api/health`
- Check firewall/network rules

### Slow predictions
- Enable GPU support: see `docker-compose.yml` GPU section
- Check system resources: `docker stats`
- Monitor ML service logs for errors

---

## Next Steps

1. ✅ ML Service ready for production
2. Deploy both services together (see docker-compose.prod.yml)
3. Configure CI/CD for automatic deployments
4. Set up monitoring and logging (Datadog, ELK, etc.)
5. Load test the full pipeline

