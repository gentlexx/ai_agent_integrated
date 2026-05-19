from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, AI Developer!"}

@app.get("/status")
def status():
    return {"status": "Your dev machine is ready!"}
 
