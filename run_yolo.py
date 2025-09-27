import os
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from supabase import create_client
import requests

# =======================
# Supabase Setup (use secrets in GitHub Actions)
# =======================
url = os.environ["SUPABASE_URL"]
key = os.environ["SUPABASE_KEY"]
supabase = create_client(url, key)

# =======================
# Load YOLO model
# =======================
model_path = "weights/best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = YOLO(model_path)

# =======================
# Run inference
# =======================
img_path = "test.jpg"  # 放一张测试图片在仓库里
results = model(img_path)

for r in results:
    im_array = r.plot()  # numpy array
    im = Image.fromarray(im_array[..., ::-1])  # 转 PIL
    im.save("prediction.jpg")  # 保存结果

print("✅ Inference complete, saved as prediction.jpg")

# =======================
# Insert into Supabase
# =======================
time = datetime.now().isoformat()

data = {
    "updated_at": time,
    "incident_type": "Pothole",
    "location": "Cyberjaya Persiaran Rimba",
    "description": "Pothole detected via YOLO",
    "reported_by": "Drone_A1",
    "status": "Resolved",
    "priority": "Medium"
}

response = supabase.table("incident_table").insert(data).execute()
print("✅ Supabase insert response:", response)

# =======================
# Trigger n8n webhook
# =======================
n8n_url = os.environ.get("N8N_WEBHOOK_URL")
if n8n_url:
    resp = requests.get(n8n_url, params={"ping": "start"})
    print("✅ n8n webhook response:", resp.status_code, resp.text)
else:
    print("⚠️ No N8N webhook URL set, skipped.")
