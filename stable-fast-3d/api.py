import tempfile
import logging
import asyncio
from typing import Optional
from pathlib import Path
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import trimesh
from scipy.spatial.transform import Rotation

from sf3d.system import SF3D
import sf3d.utils as sf3d_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enforce_symmetry(mesh: trimesh.Trimesh, axis=0) -> trimesh.Trimesh:
    """Enforce symmetry along specified axis (0=X, 1=Y, 2=Z) - Optimized version"""
    # Get vertices and create mirrored copy
    vertices = mesh.vertices.copy()
    
    # Use numpy operations instead of loops for better performance
    vertices[:, axis] *= -1
    mirrored = vertices.copy()
    
    # Average the original and mirrored vertices
    vertices[:, axis] = 0  # Zero out the axis we're symmetrizing
    vertices = (vertices + mirrored) / 2
    
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

def align_to_ground(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Align mesh so bottom is flat - Simplified version"""
    # Just translate to ground
    translation = [0, -mesh.bounds[0][1], 0]
    mesh.apply_translation(translation)
    return mesh

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
    # Force CPU for Mac
    device = "cpu"  # Changed from sf3d_utils.get_device()
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
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

@app.post("/generate_mesh")
async def generate_mesh(image: UploadFile = File(...)):
    try:
        logger.info(f"Received request to generate mesh for file: {image.filename}")
        
        # Validate file type
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(
                status_code=400, 
                detail="Only PNG and JPEG images are supported"
            )
        
        # Read and process image
        logger.info("Processing image...")
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        pil_image = sf3d_utils.remove_background(pil_image)
        pil_image = sf3d_utils.resize_foreground(pil_image, 0.85)
        logger.info("Image preprocessing completed")

        # Generate mesh
        logger.info("Starting mesh generation...")
        try:
            with torch.no_grad():
                mesh, _ = model.run_image(
                    pil_image,
                    bake_resolution=512,
                    remesh="none",  # Skip remeshing entirely for speed
                    vertex_count=-1,
                )
            logger.info("Initial mesh generation completed")
            
            # Quick post-processing
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            
            mesh = enforce_symmetry(mesh, axis=0)  # Optimized symmetry
            mesh = align_to_ground(mesh)  # Simplified alignment
            
        except Exception as e:
            logger.error(f"Error during mesh generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Mesh generation failed: {str(e)}")

        # Return the generated mesh
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as temp_file:
            mesh.export(temp_file.name, include_normals=True)
            return FileResponse(
                temp_file.name,
                media_type="model/gltf-binary",
                filename=f"{image.filename.rsplit('.', 1)[0]}.glb"
            )
            
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