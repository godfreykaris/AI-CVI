import asyncio
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from pathlib import Path
import pandas as pd
import httpx
import time

from typing import List, Dict, Union
from pydantic import BaseModel

from modules.dependencies import get_model_instance, get_data_preprocessor_instance, get_ai_instance, UI_URL

router = APIRouter()

model = get_model_instance()
data_preprocessor = get_data_preprocessor_instance()
ai = get_ai_instance(model=model)

# Global variable to keep track of the number of files sent for processing
files_sent_count = 0

@router.post("/process_file")
async def process_file(file_data: dict):
    global files_sent_count

    try:
        file_path = file_data.get("file_path")
        
        if not file_path:
            raise HTTPException(status_code=400, detail="File path not provided in the request body")
        
        encoded_sentences_tn, _ = data_preprocessor.process_dataset(file_path)
        predictions = ai.perform_inference(encoded_sentences_tn)
        vulnerability_predictions = predictions.tolist()

        df = pd.read_csv(file_path, sep='~', header=None, names=['Line', 'Code', 'File'])

        result = [{"Line": line,"File": file, "Code": code, "VulnerabilityPrediction": prediction} for line, file, code, prediction in zip(df['Line'], df['File'], df['Code'], vulnerability_predictions)]

        await send_result_to_ui(result)
        time.sleep(2)
	
         # Increment the global variable after processing a file
        files_sent_count += 1

        return {"results": "File Processed"}
    
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
        response = await client.post(f"{UI_URL}/ReceiveDataFromAI", json={"data": result})

    if response.status_code != 200:
        raise UISendError(response.status_code, response.text)

@router.post("/start_processing")
async def start_processing(request_data: dict):
    global files_sent_count
    
    files_sent_count = 0

    return {"message": "Started"}

@router.post("/report_task_complete")
async def report_task_complete(request_data: dict):
    global files_sent_count
    
    total_files = request_data.get("total_files", 0)
    
    while files_sent_count < total_files:
        # Sleep for 5 seconds before checking again
        await asyncio.sleep(5)
        print("Waiting")

    print(f"Done S: {files_sent_count} T: {total_files}")
    
    await send_job_complete_to_ui()

    return {"message": "Job complete"}

async def send_job_complete_to_ui():
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{UI_URL}/ReportWorkComplete")

    if response.status_code != 200:
        raise UISendError(response.status_code, response.text)

@router.post("/display_result/")
async def display_result(request: ResultRequest):
    data = request.data
    print("Data sent")
    return {"data": data}

@router.get("/")
def read_root():
    return {"Hello": "World"}
