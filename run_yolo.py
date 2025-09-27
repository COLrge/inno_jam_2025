import os
import requests
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from supabase import create_client
import argparse

def download_from_gdrive(file_id: str, dest_path: str):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    if response.status_code != 200:
        print(f"❌ 无法下载文件，HTTP 状态码: {response.status_code}")
        return

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={"id": file_id, "confirm": token}, stream=True)

    if response.status_code != 200:
        print(f"❌ 下载确认阶段失败，HTTP 状态码: {response.status_code}")
        return

    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
        print(f"✅ 文件已成功下载到 {dest_path}")
    except Exception as e:
        print(f"❌ 下载文件时出错: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)  # 删除部分下载的文件

def main(weights_path, gdrive_file_id="11-HcwFwueby8S_QjaFEUucUw9odDrQpn"):
    if not os.path.exists(weights_path):
        print(f"模型 {weights_path} 不存在，从 Google Drive 下载...")
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        download_from_gdrive(gdrive_file_id, weights_path)
        print("✅ 下载完成。")

    # 加载 YOLO 模型
    model = YOLO(weights_path)
    results = model("test.jpg")
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save("prediction.jpg")
    print("✅ 推理完成，prediction.jpg 已保存")

    # Supabase 插入数据
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    data = {
        "updated_at": datetime.now().isoformat(),
        "incident_type": "Pothole",
        "location": "Cyberjaya Persiaran Rimba",
        "description": "Pothole detected via YOLO",
        "reported_by": "Drone_A1",
        "status": "Resolved",
        "priority": "Medium"
    }
    response = supabase.table("incident_table").insert(data).execute()
    print("✅ Supabase 插入响应：", response)

    # n8n Webhook
    if "N8N_WEBHOOK_URL" in os.environ:
        resp = requests.get(os.environ["N8N_WEBHOOK_URL"], params={"ping": "start"})
        print("✅ n8n 响应：", resp.status_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="模型路径")
    parser.add_argument("--gdrive_id", type=str, required=True, help="Google Drive 文件 ID")
    args = parser.parse_args()
    main(args.weights, args.gdrive_id)
