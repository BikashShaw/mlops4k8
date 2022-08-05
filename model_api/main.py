import os
import sys
from typing import List, Any

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.logger import logger as log
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import RedirectResponse


app = FastAPI()


@app.post("/predict", response_model=ModelRequest)
async def predict(data_list: ModelResponse = None):
    pass
