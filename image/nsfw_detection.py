from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from transformers import pipeline
import glob
import os

load_directory = "./nsfw_model"
UPLOAD_DIR = "uploaded_images"
# 모델과 프로세서 불러오기
processor = AutoImageProcessor.from_pretrained(load_directory)
model = AutoModelForImageClassification.from_pretrained(load_directory)

image_files = glob.glob(os.path.join(UPLOAD_DIR, "*.*"))
image_path = image_files[0]
image = Image.open(f'{image_path}').convert('RGB')
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
print(classifier(image))