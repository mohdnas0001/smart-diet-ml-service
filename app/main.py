from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils.logger import logger
from app.models.detector import FoodDetector
from app.models.classifier import FoodClassifier
from app.models.portion_estimator import PortionEstimator
from app.services.nutrient_service import NutrientService
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.food_mapper import FoodMapper
from app.services.usda_client import USDAClient
from app.services.nutritionix_client import NutritionixClient
from app.routes import predict, health, nutrients
from app.schemas.response import ErrorResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Smart Diet ML Service v%s", settings.APP_VERSION)

    categories_path = f"{settings.DATA_DIR}/food_categories.json"
    nigerian_foods_path = f"{settings.DATA_DIR}/nigerian_foods.json"
    portion_priors_path = f"{settings.DATA_DIR}/portion_priors.json"
    food_density_path = f"{settings.DATA_DIR}/food_density_table.json"

    detector = FoodDetector(
        model_path=f"{settings.MODEL_DIR}/detector.pt",
        food_categories_path=categories_path,
    )
    classifier = FoodClassifier(
        model_path=f"{settings.MODEL_DIR}/classifier.pt",
        food_categories_path=categories_path,
    )
    portion_estimator = PortionEstimator(
        portion_priors_path=portion_priors_path,
        food_density_path=food_density_path,
    )
    usda_client = USDAClient(api_key=settings.USDA_API_KEY)
    nutritionix_client = NutritionixClient(
        app_id=settings.NUTRITIONIX_APP_ID,
        app_key=settings.NUTRITIONIX_APP_KEY,
    )
    food_mapper = FoodMapper(nigerian_foods_path=nigerian_foods_path)

    nutrient_svc = NutrientService(
        nigerian_foods_path=nigerian_foods_path,
        usda_client=usda_client,
        nutritionix_client=nutritionix_client,
    )

    demo = detector.demo_mode or classifier.demo_mode or settings.DEMO_MODE
    demo_reasons = []
    if settings.DEMO_MODE:
        demo_reasons.append("DEMO_MODE=true in environment")
    if detector.demo_mode:
        detail = detector.load_error or "detector is in demo mode"
        demo_reasons.append(f"detector demo mode: {detail}")
    if classifier.demo_mode:
        detail = classifier.load_error or "classifier is in demo mode"
        demo_reasons.append(f"classifier demo mode: {detail}")

    pipeline = AnalysisPipeline(
        detector=detector,
        classifier=classifier,
        portion_estimator=portion_estimator,
        nutrient_service=nutrient_svc,
        food_mapper=food_mapper,
        demo_mode=demo,
    )

    app.state.pipeline = pipeline
    app.state.nutrient_service = nutrient_svc
    app.state.model_diagnostics = {
        "env_demo_mode": settings.DEMO_MODE,
        "pipeline_demo_mode": demo,
        "demo_reasons": demo_reasons,
        "detector": {
            "loaded": not detector.demo_mode,
            "model_path": detector.model_path,
            "load_error": detector.load_error,
        },
        "classifier": {
            "loaded": not classifier.demo_mode,
            "model_path": classifier.model_path,
            "load_error": classifier.load_error,
            "trained_classes_count": len(classifier.trained_classes),
            "classes_file_found": classifier.classes_file_found,
        },
    }

    if demo:
        logger.warning("Pipeline started in demo mode. Reasons: %s", "; ".join(demo_reasons) or "unknown")
    else:
        logger.info("Pipeline running with production models")

    logger.info("Pipeline ready. Demo mode: %s", demo)
    yield
    logger.info("Smart Diet ML Service shutting down")


app = FastAPI(
    title="Smart Diet ML Service",
    version=settings.APP_VERSION,
    description="AI-powered Food Recognition & Nutrient Estimation microservice",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    payload = ErrorResponse(
        message=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        error_code=exc.__class__.__name__,
        details=None,
    )
    return JSONResponse(content=payload.model_dump(), status_code=exc.status_code)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    payload = ErrorResponse(
        message="Internal server error",
        error_code="internal_server_error",
        details=str(exc),
    )
    return JSONResponse(content=payload.model_dump(), status_code=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(nutrients.router)


@app.get("/")
async def root():
    return {"message": "Smart Diet ML Service", "version": settings.APP_VERSION, "docs": "/docs"}
