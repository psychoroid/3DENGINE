from fastapi import FastAPI, File, UploadFile, FileResponse
from sf3d.system import SF3D
import sf3d.utils as sf3d_utils
from PIL import Image
import tempfile
import torch

app = FastAPI()

device = sf3d_utils.get_device()
model = SF3D.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.eval()
model = model.to(device)

@app.post("/generate_mesh")
async def generate_mesh(image: UploadFile = File(...)):
    pil_image = sf3d_utils.remove_background(Image.open(image.file).convert("RGBA"))
    pil_image = sf3d_utils.resize_foreground(pil_image, 0.85)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16):
            mesh, _ = model.run_image(
                pil_image,
                bake_resolution=1024,
                remesh="none",
                vertex_count=-1,
            )

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".glb")
    mesh.export(tmp_file.name, include_normals=True)

    return FileResponse(tmp_file.name, media_type="model/gltf-binary", filename="mesh.glb") 