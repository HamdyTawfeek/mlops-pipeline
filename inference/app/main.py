import logging

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model import load_model, select_model
from app.schema import MushroomFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mushroom Classification API")


@app.post("/predict")
async def predict(data: MushroomFeatures):
    try:
        feature_keys = list(data.model_dump(by_alias=True).keys())

        model_name = select_model(feature_keys)
        model = load_model(model_name)

        input_data = pd.DataFrame([data.model_dump(by_alias=True)])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return PredictionResponse(
            prediction="edible" if prediction == 1 else "poisonous",
            probability=float(probability),
        )
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Model is not available or prediction failed. Please try again later.",
        )
