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
    对比起始帧与结束帧，进行像素级微观拆解并提供复原Prompt：

    ### 1. 人物主体
    - 五官神态：脸型、眼神焦点、微表情与唇形。
    - 头发动作：发色、发丝流向与飘动弧度。
    - 手部体态：手指关节形态、微动作及骨架重心。
    - 服饰鞋饰：服装版型、面料纹理、褶皱、鞋履与配饰。

    ### 2. 微观与环境
    - 地面细节：地表材质、微小物体（碎屑/爬虫/水渍）与倒影。
    - 空气光影：光照方向、冷暖色温与漂浮尘埃。
    - 空间运镜：机位角度、景深与相机推拉摇移矢量。

    ### 3. 复原提示词
    - 【生图Prompt】：包含所有五官、服饰、环境特征的中英关键词。
    - 【生视频Prompt】：包含动态过渡、微运动与运镜轨迹的中英文指令。
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
        }],
        temperature=0.1
    )
    return res.choices[0].message.content

if __name__ == "__main__":
    if os.path.exists("test.mp4"):
        s, e = extract_frames("test.mp4")
        result = analyze(s, e)
        print("\n" + "="*20 + " 像素级拆解报告 " + "="*20)
        print(result)
        with open("report.md", "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print("未检测到 test.mp4 文件，请上传视频后重新运行。")
