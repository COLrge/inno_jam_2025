import argparse
from ultralytics import YOLO
from PIL import Image
from supabase import create_client
from datetime import datetime
import requests

# Supabase 配置
SUPABASE_URL = "https://isguhazdbdzzxmtjhaqq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlzZ3VoYXpkYmR6enhtdGpoYXFxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTg5NTQxMDIsImV4cCI6MjA3NDUzMDEwMn0.C3io-J6eLruy3o0QfX3Bah8TRViCR__YbOqVfFnjH-w"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# n8n Webhook
N8N_URL = "https://noisseur.app.n8n.cloud/webhook/70eeb6c5-7c4d-483b-a262-5b99cc8b64a4"

def main(weights):
    # 加载模型
    print(f"✅ 使用模型: {weights}")
    model = YOLO(weights)

    # 推理示例图片
    image_path = "data/test.jpg"  # 你可以替换成实际图片路径
    results = model(image_path)

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save("prediction.jpg")

    # 上传到 Supabase
    time = datetime.utcnow().isoformat()
    data = {
        "updated_at": time,
        "incident_type": "Pothole",
        "location": "Cyberjaya Persiaran Rimba",
        "description": "Pothole detected automatically",
        "reported_by": "Drone_A1",
        "status": "Detected",
        "priority": "Medium"
    }
    response = supabase.table("incident_table").insert(data).execute()
    print("✅ Supabase response:", response)

    # 通知 n8n
    resp = requests.get(N8N_URL, params={"ping": "start"})
    print("✅ n8n response:", resp.status_code, resp.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="模型路径")
    args = parser.parse_args()
    main(args.weights)
