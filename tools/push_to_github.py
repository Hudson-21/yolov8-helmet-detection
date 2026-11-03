"""通过 GitHub Git Data API 上传仓库内容（绕过 github.com 的 git 协议）。

背景：本机网络对 github.com:443 直连与代理都不通（超时/502），但 api.github.com 正常。
git push 依赖 github.com，因此改用 REST API：
    blobs → tree → commit → 创建 refs/heads/main
效果与一次普通推送等价，只是全程走 api 域名。
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(r"E:/YOLO/safety_helmet")
OWNER, NAME = "Hudson-21", "yolov8-helmet-detection"
API = "https://api.github.com"
BRANCH = "main"
MESSAGE = """feat: 基于 YOLOv8 的安全帽佩戴检测系统

- 数据：公开数据集二次加工，19441 张 / 55381 框，8:1:1 重划分，标注校验零错误
- 模型：YOLOv8n 头部级二分类（hardhat / no-hardhat），含 CBAM、SE、P2 三组改进尝试与消融
- 功能：检测 + BoT-SORT 跟踪去重计数 + 告警冷却 + 图片/视频/摄像头三路输入
- 界面：tkinter 实时可视化，工作线程取帧、主线程刷新
- 工程：数据清洗、训练（支持断点续训）、评估、推理、导出全链路脚本化
- 实测：mAP@0.5 0.9354，纯检测 42.9 FPS，视频端到端 19.5 FPS

注：三组结构改进均未超过基线（详见 README 消融表），最终部署原生 YOLOv8n。"""

# 不走系统代理：本机代理对 github.com 返回 502，api 域名反而直连正常
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def token() -> str:
    r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("无法获取 gh token")
    return r.stdout.strip()


TOKEN = token()


def refresh_token() -> None:
    """gh 的 keyring token 偶尔会中途失效（实测上传到一半返回 401），重新取一次。"""
    global TOKEN
    TOKEN = token()


def call(method: str, path: str, payload: dict | None = None, retries: int = 4) -> dict:
    url = API + path
    body = json.dumps(payload).encode() if payload is not None else None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Authorization", f"Bearer {TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("Content-Type", "application/json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with OPENER.open(req, timeout=180) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            if e.code == 401 and attempt < retries:
                print("  token 失效，重新获取后重试 …")
                refresh_token()
                time.sleep(2)
                continue
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(3 * attempt)
                continue
            raise SystemExit(f"[{method} {path}] HTTP {e.code}: {detail}")
        except Exception as e:  # noqa: BLE001
            if attempt < retries:
                time.sleep(3 * attempt)
                continue
            raise SystemExit(f"[{method} {path}] {type(e).__name__}: {e}")
    raise SystemExit("unreachable")


def main() -> None:
    files = subprocess.run(
        ["git", "-c", "core.quotepath=false", "ls-files"],
        cwd=REPO, capture_output=True, text=True, encoding="utf-8",
    ).stdout.splitlines()
    files = [f for f in files if f.strip()]
    print(f"待上传文件：{len(files)} 个")

    # Git Data API 在"完全空的仓库"上会返回 409（必须先有提交才能建 blob）。
    # 因此先探测 main 是否存在；不存在才用 Contents API 建占位文件初始化。
    try:
        call("GET", f"/repos/{OWNER}/{NAME}/git/ref/heads/{BRANCH}")
        print(f"仓库已有 {BRANCH} 分支，跳过初始化")
    except SystemExit:
        print("初始化空仓库 …")
        call("PUT", f"/repos/{OWNER}/{NAME}/contents/.gitkeep",
             {"message": "chore: 初始化仓库", "content": base64.b64encode(b"init").decode(), "branch": BRANCH})

    tree = []
    total = 0
    for i, rel in enumerate(files, 1):
        p = REPO / rel
        data = p.read_bytes()
        total += len(data)
        blob = call(
            "POST", f"/repos/{OWNER}/{NAME}/git/blobs",
            {"content": base64.b64encode(data).decode(), "encoding": "base64"},
        )
        tree.append({"path": rel, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} 已上传 ({total / 1048576:.1f} MB)")

    print("创建目录树 …")
    t = call("POST", f"/repos/{OWNER}/{NAME}/git/trees", {"tree": tree})
    print("创建提交 …")
    c = call("POST", f"/repos/{OWNER}/{NAME}/git/commits",
             {"message": MESSAGE, "tree": t["sha"], "parents": []})
    print(f"  提交 {c['sha'][:10]}")
    print(f"更新分支 {BRANCH}（覆盖初始化提交，保持单次提交干净）…")
    call("PATCH", f"/repos/{OWNER}/{NAME}/git/refs/heads/{BRANCH}", {"sha": c["sha"], "force": True})
    print("完成")
    print(f"  仓库地址：https://github.com/{OWNER}/{NAME}")


if __name__ == "__main__":
    main()
