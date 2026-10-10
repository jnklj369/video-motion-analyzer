# -*- coding: utf-8 -*-
"""
视频运镜分析 v3
架构:代码实测一切动态(运镜/转场/光影/人物动态),大模型只看图描述静态内容。
用法: python main_v3.py (需环境变量 ARK_API_KEY / ARK_ENDPOINT_ID,视频 test.mp4 放同目录)
依赖: pip install opencv-python-headless openai
"""
import os
import cv2
import base64
import math
import numpy as np
from openai import OpenAI

API_KEY = os.getenv("ARK_API_KEY")
ENDPOINT_ID = os.getenv("ARK_ENDPOINT_ID")
VIDEO = "test.mp4"

client = OpenAI(api_key=API_KEY, base_url="https://ark.cn-beijing.volces.com/api/v3")


# ================= 第一层:代码实测 =================

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
    # 角点 + LK光流 + RANSAC,提纯镜头运动(前景物体会当outlier剔掉)
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


def classify_motion(m, w, h, dt):
    # 实测参数 -> 运镜关键词。位移按"每秒"归一化。
    if m is None or dt <= 0:
        return "未知"
    # 内点率过低:主体运动掩盖镜头运动,不硬猜
    if m["inlier_ratio"] < 0.35:
        return "主体运动主导,镜头运动不可靠"
    ndx, ndy = m["dx"] / w / dt, m["dy"] / h / dt
    ds = (m["scale"] - 1.0) / dt
    rot = abs(m["rot"]) / dt
    mag = math.hypot(ndx, ndy)
    tags = []
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


def motion_timeline(samples, fps):
    # 输出运镜时间线: [(开始秒, 结束秒, 关键词), ...]
    if len(samples) < 2:
        return []
    h, w = samples[0][1].shape[:2]
    tl = []
    pi, pg = samples[0][0], cv2.cvtColor(samples[0][1], cv2.COLOR_BGR2GRAY)
    for idx, f in samples[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        dt = (idx - pi) / fps
        tl.append((pi / fps, idx / fps,
                   classify_motion(estimate_global_motion(pg, g), w, h, dt)))
        pi, pg = idx, g
    return tl


def detect_transitions(samples, fps):
    # 相邻采样帧直方图相关:剧变=硬切。返回 [(开始秒, 结束秒, 类型), ...]
    cuts = []
    for k in range(1, len(samples)):
        a = cv2.cvtColor(samples[k-1][1], cv2.COLOR_BGR2HSV)
        b = cv2.cvtColor(samples[k][1], cv2.COLOR_BGR2HSV)
        ha = cv2.calcHist([a], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hb = cv2.calcHist([b], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(ha, ha)
        cv2.normalize(hb, hb)
        corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
        s, e = samples[k-1][0] / fps, samples[k][0] / fps
        if corr < 0.45:
            cuts.append((s, e, "硬切"))
    return cuts


def lighting_curve(samples, fps):
    # 每帧:平均亮度(0-255) + 冷暖(>0偏暖,<0偏冷)。返回 ([(秒,亮度,冷暖)], 趋势描述)
    pts = []
    for idx, f in samples:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        pts.append((idx / fps, float(hsv[:, :, 2].mean()),
                    float(lab[:, :, 2].mean()) - 128))
    if len(pts) < 2:
        return pts, "数据不足"
    # 趋势:前半 vs 后半均值
    mid = len(pts) // 2
    b0 = sum(p[1] for p in pts[:mid]) / mid
    b1 = sum(p[1] for p in pts[mid:]) / (len(pts) - mid)
    w0 = sum(p[2] for p in pts[:mid]) / mid
    w1 = sum(p[2] for p in pts[mid:]) / (len(pts) - mid)
    db = b1 - b0
    trend = "亮度" + ("明显变亮" if db > 25 else ("明显变暗" if db < -25 else "基本稳定"))
    dw = w1 - w0
    trend += ",色温" + ("转暖" if dw > 8 else ("转冷" if dw < -8 else "稳定"))
    # 骤变点
    for k in range(1, len(pts)):
        if abs(pts[k][1] - pts[k-1][1]) > 40:
            trend += f",{pts[k][0]:.1f}s处亮度骤变({pts[k-1][1]:.0f}->{pts[k][1]:.0f})"
            break
    return pts, trend


def motion_regions(samples, fps):
    # 帧差法找运动区域:中心轨迹 + 面积占比 + 强度分级
    # 返回 [(开始秒, 结束秒, 中心x, 中心y, 面积%, 强度), ...]
    out = []
    prev_g = cv2.cvtColor(samples[0][1], cv2.COLOR_BGR2GRAY)
    for k in range(1, len(samples)):
        g = cv2.cvtColor(samples[k][1], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_g, g)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ys, xs = np.where(th > 0)
        s, e = samples[k-1][0] / fps, samples[k][0] / fps
        h, w = th.shape
        if len(xs) == 0:
            out.append((s, e, 0, 0, 0, "静止"))
        else:
            area = len(xs) / th.size * 100
            inten = "大幅运动" if area > 30 else ("中幅运动" if area > 10 else "微动")
            out.append((s, e, float(xs.mean() / w), float(ys.mean() / h),
                        round(area, 1), inten))
        prev_g = g
    return out


# ================= 第二层:大模型只看静态内容 =================

def b64(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def fmt_timeline(tl):
    return "\n".join(" - %.1fs -> %.1fs: %s" % (s, e, kw) for s, e, kw in tl)


def llm_report(keyframes, tl_motion, tl_trans, light_trend, tl_person):
    motion_text = fmt_timeline(tl_motion)
    # 生视频Prompt只用可靠的运镜段
    clean = [(s, e, kw) for s, e, kw in tl_motion if "不可靠" not in kw]
    clean_text = fmt_timeline(clean) if clean else "(无可靠运镜段,按固定机位处理)"
    trans_text = fmt_timeline(tl_trans) if tl_trans else "(无硬切,全片连续)"
    person_text = "\n".join(
        " - %.1fs -> %.1fs: %s(运动面积%s%%,中心%s)" % (
            s, e, inten, ap, "(%.2f,%.2f)" % (cx, cy) if ap else "")
        for s, e, cx, cy, ap, inten in tl_person)
    prompt = (
        "你是视频分镜师。下方【】内均为代码实测的真实数据,请严格采用,不要编造。\n\n"
        "【运镜时间线(实测)】\n" + motion_text + "\n\n"
        "【转场点(实测)】\n" + trans_text + "\n\n"
        "【光影变化趋势(实测)】\n" + light_trend + "\n\n"
        "【人物动态强度(实测,帧差法)】\n" + person_text + "\n\n"
        "请看关键帧图片(按时间顺序),输出:\n"
        "1. 【静态拆解】人物五官/服饰/场景/光影关键词(中英)。\n"
        "2. 【人物动作】只看关键帧,描述人物动作姿势随时间的变化"
        "(如:行走/转身/面向镜头/抬手),中英双语。\n"
        "3. 【生图Prompt】首帧复原用,中英关键词。\n"
        "4. 【生视频Prompt】综合可靠运镜段+转场点+光影趋势+人物动作,"
        "写成连贯分镜指令,中英双语。转场处注明硬切。\n"
        "可靠运镜段:\n" + clean_text + "\n"
    )
    content = [{"type": "text", "text": prompt}]
    for _, img in keyframes:
        content.append({"type": "image_url", "image_url":
                        {"url": "data:image/jpeg;base64," + b64(img)}})
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

    samples, fps = sample_frames(VIDEO, n=10)

    tl_motion = motion_timeline(samples, fps)
    tl_trans = detect_transitions(samples, fps)
    light_pts, light_trend = lighting_curve(samples, fps)
    tl_person = motion_regions(samples, fps)

    print("=" * 20 + " 实测运镜时间线 " + "=" * 20)
    print(fmt_timeline(tl_motion))
    print("\n" + "=" * 20 + " 转场点 " + "=" * 20)
    print(fmt_timeline(tl_trans) if tl_trans else "(无)")
    print("\n" + "=" * 20 + " 光影趋势 " + "=" * 20)
    print(light_trend)
    print("\n" + "=" * 20 + " 人物动态 " + "=" * 20)
    for s, e, cx, cy, ap, inten in tl_person:
        print("%.1fs -> %.1fs: %s (面积%s%%)" % (s, e, inten, ap))

    kf_idx = sorted({0, len(samples) // 4, len(samples) // 2,
                     3 * len(samples) // 4, len(samples) - 1})
    keyframes = [samples[i] for i in kf_idx if i < len(samples)]
    for n, (_, img) in enumerate(keyframes):
        cv2.imwrite("kf_%d.jpg" % n, img)

    result = llm_report(keyframes, tl_motion, tl_trans, light_trend, tl_person)
    print("\n" + "=" * 20 + " 复原提示词报告 " + "=" * 20)
    print(result)
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("【实测运镜时间线】\n" + fmt_timeline(tl_motion) + "\n\n")
        f.write("【转场点】\n" +
                (fmt_timeline(tl_trans) if tl_trans else "(无)") + "\n\n")
        f.write("【光影趋势】\n" + light_trend + "\n\n")
        f.write("【人物动态】\n")
        for s, e, cx, cy, ap, inten in tl_person:
            f.write("%.1fs -> %.1fs: %s (面积%s%%)\n" % (s, e, inten, ap))
        f.write("\n" + result)
    print("\n已保存: report.txt + kf_0..4.jpg")
