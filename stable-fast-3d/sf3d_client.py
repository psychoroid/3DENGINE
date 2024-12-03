import requests

def generate_mesh(image_path):
    url = "http://localhost:8000/generate_mesh"
    
    with open(image_path, "rb") as f:
        files = {"image": f}
        response = requests.post(url, files=files)
    
    if response.status_code == 200:
        with open("output_mesh.glb", "wb") as f:
            f.write(response.content)
        print("Mesh generated successfully. Saved as output_mesh.glb")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    image_path = "path/to/your/image.png"
    generate_mesh(image_path) 