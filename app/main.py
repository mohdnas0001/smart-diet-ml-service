from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Module-level singletons (populated on startup)
pipeline: AnalysisPipeline | None = None
nutrient_service: NutrientService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, nutrient_service
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

    nutrient_service = NutrientService(
        nigerian_foods_path=nigerian_foods_path,
        usda_client=usda_client,
        nutritionix_client=nutritionix_client,
    )

    demo = detector.demo_mode or classifier.demo_mode or settings.DEMO_MODE
    pipeline = AnalysisPipeline(
        detector=detector,
        classifier=classifier,
        portion_estimator=portion_estimator,
        nutrient_service=nutrient_service,
        food_mapper=food_mapper,
        demo_mode=demo,
    )

    logger.info("Pipeline ready. Demo mode: %s", demo)
    yield
    logger.info("Smart Diet ML Service shutting down")


app = FastAPI(
    title="Smart Diet ML Service",
    version=settings.APP_VERSION,
    description="AI-powered Food Recognition & Nutrient Estimation microservice",
    lifespan=lifespan,
)

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
