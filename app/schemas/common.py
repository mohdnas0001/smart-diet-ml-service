from enum import Enum


class FoodRegion(str, Enum):
    nigerian = "nigerian"
    international = "international"


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"
    unknown = "unknown"
