from pydantic import BaseModel
from typing import Optional
from app.schemas.common import MealType


class AnalysisRequest(BaseModel):
    meal_type: Optional[MealType] = MealType.unknown
    user_id: Optional[str] = None
