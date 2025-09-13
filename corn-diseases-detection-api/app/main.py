from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.infer import predict

app = FastAPI(title="Maize Disease API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Sube una imagen"})
    content = await file.read()
    out = predict(content)
    return {"prediction": out}
