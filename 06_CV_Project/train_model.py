import cv2
import os
import numpy as np

def train_model(data_dir: str = 'dataset', model_path: str = 'face_model.yml'):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels = [], []
    
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            img_path = os.path.join(label_dir, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(int(label))
            
    recognizer.train(faces, np.array(labels))
    recognizer.write(model_path)
    print(f"학습 완료, 모델을 '{model_path}'에 저장했습니다.")
    
if __name__ == "__main__":
    train_model()