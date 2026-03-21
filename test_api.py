import requests

url = "http://localhost:8000/extract"

# Use full path
image_path = r"F:\project_yolo\your_image_name.jpg"

with open(image_path, "rb") as f:
    response = requests.post(
        url,
        files={"file": ("pan.jpg", f, "image/jpeg")}
    )

print(response.json())