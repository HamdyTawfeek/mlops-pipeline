from typing import Union

from pydantic import BaseModel, Field


class NaiveBayesFeatures(BaseModel):
    cap_diameter: int = Field(alias="cap-diameter")
    cap_shape: int = Field(alias="cap-shape")
    gill_attachment: int = Field(alias="gill-attachment")
    gill_color: int = Field(alias="gill-color")

    class Config:
        populate_by_name = True


class LogisticRegressionFeatures(BaseModel):
    stem_height: float = Field(alias="stem-height")
    stem_width: float = Field(alias="stem-width")
    stem_color: float = Field(alias="stem-color")
    season: float

    class Config:
        populate_by_name = True


MushroomFeatures = Union[NaiveBayesFeatures, LogisticRegressionFeatures]


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
