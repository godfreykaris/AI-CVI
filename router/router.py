import json
from fastapi import APIRouter, Depends, HTTPException, Query
from pathlib import Path
import pandas as pd
import httpx

from typing import List, Dict, Union
from pydantic import BaseModel

from modules.dependencies import get_model_instance, get_data_preprocessor_instance, get_ai_instance, UI_URL

router = APIRouter()

model = get_model_instance()
data_preprocessor = get_data_preprocessor_instance()
ai = get_ai_instance(model=model)

def validate_file_path(file_path: str = Query(..., description="Path to the CSV file to be processed")):
    path = Path(file_path)
    if not path.is_file():
        raise HTTPException(status_code=400, detail="Invalid file path")
    return path

@router.get("/process_file/")
async def process_file(file_path: Path = Depends(validate_file_path)):
    try:
        encoded_sentences_tn, _ = data_preprocessor.process_dataset(file_path)
        predictions = ai.perform_inference(encoded_sentences_tn)
        vulnerability_predictions = predictions.tolist()

        df = pd.read_csv(file_path, sep='~', header=None, names=['Line', 'Code', 'File'])

        result = [{"Line": line,"File": file, "Code": code, "VulnerabilityPrediction": prediction} for line, file, code, prediction in zip(df['Line'], df['File'], df['Code'], vulnerability_predictions)]

        await send_result_to_ui(result)

        return {"results": result}
    
    except UISendError as e:       
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
class ResultRequest(BaseModel):
    data: List[Dict[str, Union[str, int]]]

class UISendError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Error sending result to UI service: {detail}")

async def send_result_to_ui(result: List[Dict[str, Union[str, int]]]):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{UI_URL}/display_result/", json={"data": result})

    if response.status_code != 200:
        raise UISendError(response.status_code, response.text)

@router.post("/display_result/")
async def display_result(request: ResultRequest):
    data = request.data
    print(data)
    return {"data": data}

@router.get("/")
def read_root():
    return {"Hello": "World"}
