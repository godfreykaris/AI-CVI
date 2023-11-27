from fastapi import FastAPI
from fastapi import Depends

from router.router import router as main_router

app = FastAPI()

app.include_router(main_router, prefix="/api")
