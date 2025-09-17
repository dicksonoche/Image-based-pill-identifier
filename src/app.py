from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms, models
import io
from PIL import Image
import cv2
import numpy as np

app = FastAPI()

# Load model
num_classes = 5
classes = ['aspirin', 'ibuprofen', 'acetaminophen', 'atorvastatin', 'metformin']
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('pill_model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        pred_class = classes[predicted.item()]
        confidence_score = confidence.item() * 100

    # Human-in-loop: If low confidence, add overlay
    if confidence_score < 80:
        # Simple overlay: Draw rectangle (assume pill is centered)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w = img_cv.shape[:2]
        cv2.rectangle(img_cv, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 255), 2)  # Red box
        overlay_img = cv2.imencode('.jpg', img_cv)[1].tobytes()
        return {"prediction": pred_class, "confidence": confidence_score, "message": "Human review required", "overlay_image": overlay_img}
    else:
        return {"prediction": pred_class, "confidence": confidence_score, "message": "Confirmed"}

# Run with: uvicorn app:app --reload