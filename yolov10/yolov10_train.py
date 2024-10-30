from ultralytics import YOLOv10
import os

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == '__main__':
    model = YOLOv10("ultralytics/cfg/models/v10/yolov10s.yaml")

results = model.train(data="/root/zz/zz.yaml", epochs=100, batch=16, workers=8)

