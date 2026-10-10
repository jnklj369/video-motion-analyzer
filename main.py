# -*- coding: utf-8 -*-
"""
视频运镜分析 v2
原理:代码用光流实测镜头运动(准),大模型只看图描述内容(稳),各干各的。
用法: python main.py  (需环境变量 ARK_API_KEY / ARK_ENDPOINT_ID,视频 test.mp4 放同目录)
依赖: pip install opencv-python-headless openai
"""
import os
import cv2
import base64
import math
from openai import OpenAI

# ============ 配置:全部走环境变量,不要硬编码密钥 ============
API_KEY = os.getenv("ARK_API_KEY")
ENDPOINT_ID = os.getenv("ARK_ENDPOINT_ID")
VIDEO = "test.mp4"

client = OpenAI(api_key=API_KEY, base_url="https://ark.cn-beijing.volces.com/api/v3")


# ============ 第一层:代码实测镜头运动 ============

def sample_frames(video_path, n=10):
    # 沿时间轴均匀抽 n 帧,返回 ([(帧号, 图像), ...], fps)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total < 2:
        cap.release()
        return [], fps
    idxs = sorted({int(round(i * (total - 1) / (n - 1))) for i in range(n)})
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            out.append((i, f))
    cap.release()
    return out, fps


def estimate_global_motion(prev_gray, gray):
    # 角点 + LK光流 + RANSAC,提纯出镜头运动
    # RANSAC 会把前景运动物体当 outlier 自动剔掉,留下占主导的镜头运动
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300,
                                 qualityLevel=0.01, minDistance=8)
    if p0 is None:
        return None
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    m = (st == 1).reshape(-1)
    q0 = p0[m].reshape(-1, 2)
    q1 = p1[m].reshape(-1, 2)
    if len(q0) < 12:
        return None
    M, inliers = cv2.estimateAffinePartial2D(
        q0, q1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    return {
        "dx": float(M[0, 2]),
        "dy": float(M[1, 2]),
        "rot": math.degrees(math.atan2(M[1, 0], M[0, 0])),
        "scale": math.sqrt(float(M[0, 0]) ** 2 + float(M[1, 0]) ** 2),
        "inlier_ratio": float(inliers.sum()) / len(q0),
    }


def classify(m, w, h, dt):
    # 实测参数 -> 运镜关键词。位移按"每秒"归一化,不同采样间隔可比。
    # 阈值按 720p/1080p 常规运镜调过,太灵敏/太迟钝就调这几个数。
    if m is None or dt <= 0:
        return "未知"
    ndx, ndy = m["dx"] / w / dt, m["dy"] / h / dt
    ds = (m["scale"] - 1.0) / dt
    rot = abs(m["rot"]) / dt
    mag = math.hypot(ndx, ndy)
    tags = []
    # 内点率过低:主体运动掩盖了镜头运动,硬猜方向只会错,直接报不可靠
    if m["inlier_ratio"] < 0.35:
        return "主体运动主导,镜头运动不可靠"
    if rot > 3.0:
        tags.append("旋转镜头roll")
    if ds > 0.008:
        tags.append("推镜push-in")
    elif ds < -0.008:
        tags.append("拉镜pull-out")
    if mag > 0.008:
        if abs(ndx) >= abs(ndy):
            tags.append("右摇pan-right" if ndx > 0 else "左摇pan-left")
        else:
            tags.append("上摇tilt-up" if ndy < 0 else "下摇tilt-down")
        tags.append("快速" if mag > 0.06 else ("中速" if mag > 0.02 else "慢速"))
    if not tags:
        tags.append("固定机位static")
    return " + ".join(tags)


def motion_timeline(video_path, n=10):
    # 输出整条运镜时间线: [(开始秒, 结束秒, 运镜关键词), ...]
    samples, fps = sample_frames(video_path, n)
    if len(samples) < 2:
        return [], samples, fps
    h, w = samples[0][1].shape[:2]
    tl = []
    pi, pg = samples[0][0], cv2.cvtColor(samples[0][1], cv2.COLOR_BGR2GRAY)
    for idx, f in samples[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        dt = (idx - pi) / fps
        tl.append((pi / fps, idx / fps,
                   classify(estimate_global_motion(pg, g), w, h, dt)))
        pi, pg = idx, g
    return tl, samples, fps


# ============ 第二层:大模型只负责"看内容",运动参数由代码喂给它 ============

def b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def llm_report(keyframes, timeline):
    clean = [(s, e, kw) for s, e, kw in timeline if "不可靠" not in kw]
    if not clean:
        clean_note = "注:本视频以主体运动为主,未检测到明确镜头运动,请按固定机位处理。"
    else:
        clean_note = ""
    motion_text = "\n".join(
        " - %.1fs -> %.1fs: %s" % (s, e, kw) for s, e, kw in clean)
    if clean_note:
        motion_text += "\n" + clean_note
    prompt = (
        "你是视频分镜师。下方【运镜时间线】是光流算法逐段实测的"
        "真实镜头运动数据,请严格采用,不要自己编造运镜。\n\n"
        "【运镜时间线】\n" + motion_text + "\n\n"
        "请看关键帧图片(按时间顺序),输出:\n"
        "1. 【静态拆解】人物五官/服饰/场景/光影关键词(中英)。\n"
        "2. 【生图Prompt】首帧复原用,中英关键词。\n"
        "3. 【生视频Prompt】把上面的运镜时间线翻译成连贯运镜指令,"
        "中英双语,时间分段与实测一致。\n"
    )
    content = [{"type": "text", "text": prompt}]
    for _, img in keyframes:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64," + b64(img)}})
    res = client.chat.completions.create(
        model=ENDPOINT_ID,
        messages=[{"role": "user", "content": content}],
        temperature=0.1,
    )
    return res.choices[0].message.content


if __name__ == "__main__":
    if not os.path.exists(VIDEO):
        print("未检测到 %s,请上传视频后重新运行。" % VIDEO)
        raise SystemExit
    if not API_KEY or not ENDPOINT_ID:
        print("缺少环境变量 ARK_API_KEY / ARK_ENDPOINT_ID")
        raise SystemExit

    timeline, samples, fps = motion_timeline(VIDEO, n=10)
    print("=" * 20 + " 实测运镜时间线 " + "=" * 20)
    for s, e, kw in timeline:
        print("%5.1fs -> %5.1fs : %s" % (s, e, kw))

    # 关键帧:首/1/4/中/3/4/尾共5帧,供大模型看内容
    kf_idx = sorted({0, len(samples) // 4, len(samples) // 2,
                     3 * len(samples) // 4, len(samples) - 1})
    keyframes = [samples[i] for i in kf_idx if i < len(samples)]
    for n, (_, img) in enumerate(keyframes):
        cv2.imwrite("kf_%d.jpg" % n, img)

    result = llm_report(keyframes, timeline)
    print("\n" + "=" * 20 + " 复原提示词报告 " + "=" * 20)
    print(result)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("【实测运镜时间线】\n")
        for s, e, kw in timeline:
            f.write("%.1fs -> %.1fs : %s\n" % (s, e, kw))
        f.write("\n" + result)
    print("\n已保存: report.txt + kf_0..4.jpg")
