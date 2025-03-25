from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import List
import os
import logging

from new import predict_custom_input
from fastapi.responses import JSONResponse


logging.basicConfig(level=logging.ERROR)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Text(BaseModel):
    userId: str
    text: str


@app.get("/")
def home():
    return Response("Welcome To Emotion System")


@app.post("/emotion")
def predict_emotion(data: Text):
    try:
        user_id = data.userId
        text = data.text

        response = predict_custom_input(text=text)

        return JSONResponse(content={"userId": user_id, "Emotion": response})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")
