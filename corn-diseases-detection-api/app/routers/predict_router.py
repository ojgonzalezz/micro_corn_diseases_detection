#####################################################################################
# ---------------------------------- Corn App Router --------------------------------
#####################################################################################

##########################
# ---- Depedendencies ----
##########################

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.pipelines.infer import predict

from src.core.load_env import EnvLoader

################
# ---- main ----
################

router = APIRouter(prefix="/api")

@router.get("/health")
def health():
    """Endpoint for a simple health check."""
    return {"status": "ok"}


@router.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Endpoint to accept an image and return a disease prediction."""
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Sube una imagen"})
    content = await file.read()
    
    try:
        out = predict(content)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    return {"prediction": out}