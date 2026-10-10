import os
import cv2
import base64
from openai import OpenAI

API_KEY = os.getenv("ARK_API_KEY")
ENDPOINT_ID = os.getenv("ARK_ENDPOINT_ID")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://ark.cn-beijing.volces.com/api/v3"
)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, start_frame = cap.read()
    cv2.imwrite("start.jpg", start_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 1))
    _, end_frame = cap.read()
    cv2.imwrite("end.jpg", end_frame)
    cap.release()
    return "start.jpg", "end.jpg"

def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def analyze(s_path, e_path):
    b64_start = to_base64(s_path)
    b64_end = to_base64(e_path)
    
    prompt = """
    对比这两张连续画面的起始帧与结束帧，详细拆解其运镜与视觉动态：
    1. 【运镜轨迹】：说明镜头移动方向（推/拉/摇/移/俯仰/环绕等）
    2. 【主体动态】：主体核心动作或状态变化
    3. 【AI复原提示词】：给出适用于可灵/Runway/即梦的中英文 Prompt
    """
    
    res = client.chat.completions.create(
        model=ENDPOINT_ID,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_start}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_end}"}}
            ]
        }]
    )
    return res.choices[0].message.content

if __name__ == "__main__":
    if os.path.exists("test.mp4"):
        s, e = extract_frames("test.mp4")
        result = analyze(s, e)
        print("\n" + "="*20 + " 运镜拆解报告 " + "="*20)
        print(result)
        # 将分析结果保存为 Markdown 文件
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print("未检测到 test.mp4 文件，请上传视频后重新运行。")
