import os
import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set environment variables
os.environ['ATTN_BACKEND'] = 'xformers'
os.environ['SPCONV_ALGO'] = 'native'

# Initialize model
try:
    logger.info("Initializing model...")
    model = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    model.cuda()
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

# Create output directory
OUTPUT_DIR = Path("output_meshes")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

@app.post("/generate_mesh")
async def generate_mesh(image: UploadFile = File(...)):
    try:
        logger.info("Received request to generate mesh")
        
        if not image or not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
            
        # Validate file type
        filename = image.filename
        if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")
        
        # Process image
        pil_image = Image.open(image.file).convert("RGBA")
        
        # Generate mesh
        logger.info("Starting mesh generation...")
        try:
            with torch.no_grad():
                outputs = model.run(
                    pil_image,
                    seed=1,
                    formats=["gaussian", "mesh"],
                )
            logger.info("Mesh generation completed successfully")
        except Exception as e:
            logger.error("Error during mesh generation: {}", str(e))
            raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")

        # Export mesh as GLB
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=0.95,
                texture_size=1024,
            )
            glb.export(temp_file.name)
            output_filename = f"{Path(filename).stem}.glb"
            return FileResponse(
                temp_file.name,
                media_type="model/gltf-binary",
                filename=output_filename
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Unexpected error: {}", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/meshes")
async def list_meshes():
    """List all generated meshes in the output directory."""
    meshes = [f.name for f in OUTPUT_DIR.glob("*.glb")]
    return {"meshes": meshes} 