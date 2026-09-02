# -*- coding: utf-8 -*-
"""长文本分段合成功能测试。

覆盖（需要服务已启动并完成模型加载）：
1. POST /api/tts/batch/split        —— 按句号切分预览
2. POST /api/tts/batch              —— 创建分段合成任务（逐句合成）
3. GET  /api/tts/batch/{task_id}    —— 轮询进度
4. GET  /api/tts/batch/{task_id}/download —— 一键下载全部段落 zip

用法（先启动服务：start.bat 或 启动.bat）：
    venv\Scripts\python.exe -X utf8 tests\batch_tts_test.py

音色：自动取第一个「自定义克隆音色」（/v1/voices/custom）。
若还没有自定义音色，请先在 Web 页面用一段 3-30 秒参考音频创建一个。

输出：tests\output\batch_test_*.zip（可删除）
"""
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://localhost:8188"
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

TEST_TEXT = (
    "你好，欢迎使用长文本分段合成。"
    "这是一段测试文本！"
    "今天天气怎么样？"
    "请多保重身体。\n"
    "第二行内容。"
    "最后一句测试。"
)
# 带对话引号的长文本（单独成行的引号不应被切成独立句子——回归测试）
QUOTE_TEXT = (
    '"喂，你在哪儿呢？\n'
    '"我已经到楼下了。\n'
    '电话那头传来朋友小王的催促声。\n'
    '"\n'
    '"行，那你快点，咱们说好了七点开场的电影，再不出发就赶不上了。\n'
    '"\n'
    '知道啦知道啦，我这不是在赶嘛。对了，今天天气有点凉，你穿厚点没有？'
)


def api_json(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))


def api_form(path, fields):
    boundary = "----dshbatchtest123"
    parts = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(
        BASE + path, data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))


def pick_voice():
    """取第一个自定义克隆音色；没有则报错退出"""
    r = api_json("/v1/voices/custom", {})
    voices = r.get("voices") or []
    if not voices:
        raise SystemExit(
            "[SKIP] no custom clone voice found. Create one on the Web page "
            "(upload a 3-30s reference audio) and rerun this test."
        )
    return voices[0]


def main():
    # 0) 找一个自定义音色（id 动态获取，避免写死本机音色）
    voice = pick_voice()
    voice_id = voice["id"]
    prompt_text = voice.get("text") or ""
    print(f"[0] using voice: {voice_id} ({voice.get('name', '')})")

    # 1) 切分预览（普通文本）
    r = api_json("/api/tts/batch/split", {"text": TEST_TEXT})
    print(f"[1] split total = {r['total']} 句")
    assert r["total"] == 6, f"expect 6, got {r['total']}"
    for s in r["sentences"]:
        print(f"    [{s['index']}] ({s['chars']}字) {s['text']}")

    # 1b) 切分预览（带引号对话，回归：不得出现纯引号片段）
    rq = api_json("/api/tts/batch/split", {"text": QUOTE_TEXT})
    print(f"[1b] quote split total = {rq['total']} 句")
    assert rq["total"] == 6, f"quote text expect 6, got {rq['total']}"
    assert all(any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in s["text"]) for s in rq["sentences"]), \
        "pure-punctuation fragment exists (quote line split wrongly)"
    for s in rq["sentences"]:
        print(f"    [{s['index']}] ({s['chars']}字) {s['text']}")

    # 2) 创建任务
    r = api_form("/api/tts/batch", {
        "text": TEST_TEXT,
        "voice_id": voice_id,
        "mode": "zero_shot",
        "prompt_text": prompt_text,
        "speed": "1.0",
        "max_len": "100",
    })
    task_id = r["task_id"]
    print(f"[2] create task = {task_id} (total={r['total']})")
    assert r["total"] == 6

    # 3) 轮询
    status = None
    for _ in range(120):
        t = json.loads(urllib.request.urlopen(
            BASE + f"/api/tts/batch/{task_id}", timeout=60
        ).read().decode("utf-8"))
        status = t["status"]
        done = sum(1 for s in t["sentences"] if s["status"] == "done")
        failed = sum(1 for s in t["sentences"] if s["status"] == "failed")
        print(f"[3] poll status={status} done={done}/{t['total']} failed={failed}")
        if status in ("completed", "failed"):
            break
        time.sleep(2)
    assert status == "completed", f"task not completed: {status}"
    assert all(s["status"] == "done" for s in t["sentences"]), "some sentences not done"

    # 4) 一键下载 zip
    data = urllib.request.urlopen(
        BASE + f"/api/tts/batch/{task_id}/download", timeout=120
    ).read()
    out = OUT / f"batch_test_{task_id}.zip"
    out.write_bytes(data)
    print(f"[4] zip downloaded {len(data)} bytes -> {out}")
    assert len(data) > 1000, "zip too small, packaging may have failed"

    print("\n[PASS] long-text batch TTS test passed")


if __name__ == "__main__":
    main()
