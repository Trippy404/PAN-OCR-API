import requests

url = "http://localhost:8000/extract"

# Use full path
image_path = r"F:\project_yolo\dataset\test_pan.jpg"

with open(image_path, "rb") as f:
    response = requests.post(
        url,
        files={"file": ("test_pan.jpg", f, "image/jpeg")}
    )

print(response.json())