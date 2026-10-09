# Eğitilen modeli test etme aşamasi

from ultralytics import YOLO
import cv2 as cv

# Görsel yolu verme
path="test1.jpg"
image=cv.imread(path)

# Model tanımı
model=YOLO("weights/best.pt")

# Tahmin al
results=model(image)

# Kutu koordinatları
for box in results[0].boxes:
    x1,y1,x2,y2=map(int,box.xyxy[0])
    cls_id=int(box.cls[0])
    confidence=float(box.conf[0])
    label=f"{model.names[cls_id]} conf: {confidence:.2f}"


# Kutu çiz

cv.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

cv.putText(image,label,(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)


# Display prediction

cv.imshow("Prediction",image)
cv.waitKey(0)
cv.destroyAllWindows()

# Save prediction

cv.imwrite("Prediction-result.jpg",image)


 