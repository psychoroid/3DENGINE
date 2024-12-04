import tempfile
import logging
import asyncio
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from sf3d.system import SF3D
import sf3d.utils as sf3d_utils

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

# Initialize device and model
try:
    logger.info("Initializing model...")
    device = sf3d_utils.get_device()
    logger.info(f"Using device: {device}")
    
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.eval()
    model = model.to(device)
    logger.info("Model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing model: {str(e)}")
    raise

# Create output directory
OUTPUT_DIR = Path("output_meshes")
OUTPUT_DIR.mkdir(exist_ok=True)

@app.post("/generate_mesh")
async def generate_mesh(image: UploadFile = File(...)):
    try:
        logger.info(f"Received request to generate mesh for file: {image.filename}")
        
        # Validate file type
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")
        
        # Process image
        logger.info("Processing image...")
        pil_image = sf3d_utils.remove_background(Image.open(image.file).convert("RGBA"))
        pil_image = sf3d_utils.resize_foreground(pil_image, 0.85)
        logger.info("Image preprocessing completed")

        # Generate mesh with timeout
        logger.info("Starting mesh generation...")
        try:
            with torch.no_grad():
                # Use autocast only for CUDA devices
                context = (
                    torch.autocast(device_type=device, dtype=torch.float16)
                    if device == "cuda"
                    else torch.autocast(device_type=device, dtype=torch.float32)
                )
                
                with context:
                    mesh, _ = model.run_image(
                        pil_image,
                        bake_resolution=512,  # Reduced resolution for faster processing
                        remesh="none",
                        vertex_count=-1,
                    )
            logger.info("Mesh generation completed successfully")
        except Exception as e:
            logger.error(f"Error during mesh generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")

        # Export mesh to output directory
        try:
            output_filename = f"{image.filename.rsplit('.', 1)[0]}.glb"
            output_path = OUTPUT_DIR / output_filename
            mesh.export(str(output_path), include_normals=True)
            logger.info(f"Mesh exported to: {output_path}")
            
            return FileResponse(
                str(output_path),
                media_type="model/gltf-binary",
                filename=output_filename
            )
        except Exception as e:
            logger.error(f"Error exporting mesh: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to export mesh: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

@app.get("/meshes")
async def list_meshes():
    """List all generated meshes in the output directory."""
    meshes = [f.name for f in OUTPUT_DIR.glob("*.glb")]
    return {"meshes": meshes}