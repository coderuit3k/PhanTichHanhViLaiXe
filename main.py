import cv2
import torch
from ViT import CustomViTFusionModel
import time
from PIL import Image
import torchvision.transforms as transforms
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomViTFusionModel(num_classes=2).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

checkpoint = torch.load("checkpoint.ckpt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

mapping = {0: "drownsy", 1: "non drownsy"}

def normalize(eda, hr, temp):
    eda = (eda - 20e-6) / (0.5 - 20e-6)
    hr = (hr - 60) / (100 - 60)
    temp = (temp - 36) / (37.5 - 36)
    
    return eda, hr, temp

faces_dir = "faces"
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")

print("Nhập dữ liệu sinh trắc học: ")
physio = list(map(float, input().split()))
if len(physio) != 3:
    raise ValueError(f"Vui lòng nhập 3 chỉ số sức khỏe !")

eda, hr, temp = physio[0], physio[1], physio[2]

cam = cv2.VideoCapture(0)
prev_frame_time = time.time()
new_frame_time = 0

while cam.isOpened():
    ret, frame = cam.read()
    
    if not ret:
        print("Finished processing or cannot read the video.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 10, minSize=(10, 10))
    
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
        roi_face = frame[y: y + h, x: x + w]
        
        cv2.imwrite(os.path.join(faces_dir, f"{idx + 1}.jpg"), roi_face)
        
        with torch.no_grad():
            image = Image.fromarray(cv2.cvtColor(roi_face, cv2.COLOR_BGR2RGB)).convert("RGB")
            transformed_image = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]

            n_eda, n_hr, n_temp = normalize(eda, hr, temp)
                
            physio = torch.tensor([n_eda, n_hr, n_temp], dtype=torch.float32).reshape(1, -1).to(device)     # [1, 3]

            output = model(transformed_image, physio)               # [1, num_classes]

            # Nếu output là logits (với CrossEntropyLoss), dùng softmax
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            accuracy = probs[0][pred_class].item()
            
        cv2.putText(frame, mapping[pred_class], (x + 75, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    fps_text = f"FPS: {int(fps)}"
    
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow("Drowniess Detection in Video with FPS", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()