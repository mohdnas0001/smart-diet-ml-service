from pydantic import BaseModel
from typing import List, Dict, Optional
from app.schemas.common import FoodRegion


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class NutrientProfile(BaseModel):
    calories: float = 0.0
    carbohydrates: float = 0.0
    protein: float = 0.0
    total_fat: float = 0.0
    saturated_fat: float = 0.0
    trans_fat: float = 0.0
    monounsaturated_fat: float = 0.0
    polyunsaturated_fat: float = 0.0
    cholesterol: float = 0.0
    dietary_fiber: float = 0.0
    total_sugars: float = 0.0
    added_sugars: float = 0.0
    sodium: float = 0.0
    potassium: float = 0.0
    calcium: float = 0.0
    iron: float = 0.0
    magnesium: float = 0.0
    phosphorus: float = 0.0
    zinc: float = 0.0
    copper: float = 0.0
    manganese: float = 0.0
    selenium: float = 0.0
    iodine: float = 0.0
    chromium: float = 0.0
    molybdenum: float = 0.0
    fluoride: float = 0.0
    vitamin_a: float = 0.0
    vitamin_c: float = 0.0
    vitamin_d: float = 0.0
    vitamin_e: float = 0.0
    vitamin_k: float = 0.0
    thiamin_b1: float = 0.0
    riboflavin_b2: float = 0.0
    niacin_b3: float = 0.0
    pantothenic_acid_b5: float = 0.0
    vitamin_b6: float = 0.0
    biotin_b7: float = 0.0
    folate_b9: float = 0.0
    vitamin_b12: float = 0.0
    choline: float = 0.0
    betaine: float = 0.0
    omega_3: float = 0.0
    omega_6: float = 0.0
    epa: float = 0.0
    dha: float = 0.0
    ala: float = 0.0
    linoleic_acid: float = 0.0
    arachidonic_acid: float = 0.0
    water: float = 0.0
    ash: float = 0.0
    caffeine: float = 0.0
    theobromine: float = 0.0
    alcohol: float = 0.0
    lycopene: float = 0.0
    lutein_zeaxanthin: float = 0.0
    beta_carotene: float = 0.0
    alpha_carotene: float = 0.0
    beta_cryptoxanthin: float = 0.0
    retinol: float = 0.0
    phytosterols: float = 0.0
    stigmasterol: float = 0.0
    campesterol: float = 0.0
    beta_sitosterol: float = 0.0
    tryptophan: float = 0.0
    threonine: float = 0.0
    isoleucine: float = 0.0
    leucine: float = 0.0
    lysine: float = 0.0
    methionine: float = 0.0
    phenylalanine: float = 0.0
    valine: float = 0.0
    histidine: float = 0.0


class MacroSummary(BaseModel):
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    total_fiber: float


class FoodItem(BaseModel):
    name: str
    confidence: float
    bounding_box: BoundingBox
    portion_grams: float
    nutrients: NutrientProfile
    food_region: FoodRegion = FoodRegion.nigerian


class AnalysisResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    analysis_id: str
    image_width: int
    image_height: int
    food_items: List[FoodItem]
    total_calories: float
    total_macronutrients: MacroSummary
    processing_time_ms: float
    model_versions: Dict[str, str]
    warnings: List[str] = []
