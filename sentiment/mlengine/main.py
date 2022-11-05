from fastapi import FastAPI
from MLEngineBert import MlEngineBert

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
