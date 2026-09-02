# -*- coding: utf-8 -*-
"""
从 Docker 镜像归档 image.tar 中提取所需文件（不依赖 Docker）。

提取内容（按层顺序应用，处理 whiteout）：
  - app/**                     -> code/            （cosyvoice 包 / pretrained_models / third_party 等）
  - opt/conda/envs/cosyvoice/conda-meta/**       -> runtime/conda-meta（仅版本参考）
  - opt/conda/envs/cosyvoice/lib/python3.10/site-packages/*.dist-info 和 *.egg-info
                                                  -> runtime/site-packages-meta（仅版本参考）

用法: py -3.14 tools\\extract_image.py
"""
import os
import sys
import json
import tarfile
import shutil

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTER_TAR = os.path.join(PROJECT, "image.tar")
DEST = os.path.join(PROJECT, "code")            # app/** 解到 code/
META_DEST = os.path.join(PROJECT, "runtime", "conda-meta")
SP_META_DEST = os.path.join(PROJECT, "runtime", "site-packages-meta")

INTERESTING_TOP = ("app/", "opt/conda/envs/cosyvoice/conda-meta/",
                   "opt/conda/envs/cosyvoice/lib/python3.10/site-packages/")


def norm(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    return name


def is_wanted(name: str) -> str:
    """返回提取目的前缀(顶层目录)，不想要返回 None"""
    n = norm(name)
    if not n or n in ("app", "opt"):
        return None
    for p in INTERESTING_TOP:
        if n.startswith(p):
            return p.split("/")[0]
    # site-packages 下只要 dist-info / egg-info / *.pth / RECORD 等小元数据
    n2 = n
    parts = n2.split("/")
    if len(parts) >= 7 and parts[0] == "opt" and parts[1] == "conda" and parts[2] == "envs" and parts[3] == "cosyvoice" and parts[4] == "lib" and parts[5] == "python3.10" and parts[6] == "site-packages":
        base = parts[7]
        if base.endswith(".dist-info") or base.endswith(".egg-info") or base.endswith(".pth"):
            return "spmeta"
    return None


def safe_del(path):
    if os.path.lexists(path):
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                os.remove(path)
            except OSError:
                pass


def extract_entry(inner: tarfile.TarFile, e: tarfile.TarInfo, dest_top: str):
    """把条目 e 提取到 dest_top 下对应路径；返回 True 表示成功/已处理"""
    name = norm(e.name)
    kind = e.type
    base = os.path.basename(name)
    dname = os.path.dirname(name)

    # whiteout
    if base == ".wh..wh..opq":
        safe_del(os.path.join(dest_top, dname))
        return True
    if base.startswith(".wh."):
        target = os.path.join(dest_top, dname, base[4:])
        safe_del(target)
        return True

    target = os.path.join(dest_top, name)
    if kind == tarfile.DIRTYPE:
        os.makedirs(target, exist_ok=True)
        return True
    if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        print(f"  [skip] {name} (type {kind})")
        return True
    if kind != tarfile.REGTYPE and kind != tarfile.ARETYPE:
        return True

    # 文件
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    if os.path.isdir(target):
        safe_del(target)
    src = inner.extractfile(e)
    if src is None:
        return True
    with open(target, "wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)
    return True


def main():
    print(f"PROJECT = {PROJECT}")
    print(f"打开外层归档 {OUTER_TAR} ...")
    outer = tarfile.open(OUTER_TAR, "r")
    manifest = json.loads(outer.extractfile("manifest.json").read())
    layers = manifest[0]["Layers"]
    print(f"共 {len(layers)} 层")

    os.makedirs(DEST, exist_ok=True)
    os.makedirs(META_DEST, exist_ok=True)
    os.makedirs(SP_META_DEST, exist_ok=True)

    extracted = {"app": 0, "conda-meta": 0, "spmeta": 0}
    for li, blob in enumerate(layers):
        digest = blob.split("/")[-1]
        print(f"[{li + 1}/{len(layers)}] 层 {digest[:12]} ...", flush=True)
        m = outer.getmember(blob)
        f = outer.extractfile(m)
        inner = tarfile.open(fileobj=f, mode="r:*")
        n = 0
        for e in inner:
            name = norm(e.name)
            if not name or name in ("app", "opt"):
                continue
            if name.startswith("app/"):
                extract_entry(inner, e, DEST)
                extracted["app"] += 1
                n += 1
            elif name.startswith("opt/conda/envs/cosyvoice/conda-meta/"):
                if not name.endswith("/") and e.type == tarfile.REGTYPE:
                    extract_entry(inner, e, META_DEST)
                    extracted["conda-meta"] += 1
            elif name.startswith("opt/conda/envs/cosyvoice/lib/python3.10/site-packages/"):
                base = os.path.basename(name)
                if base.endswith(".dist-info") or base.endswith(".egg-info") or base.endswith(".pth"):
                    extract_entry(inner, e, SP_META_DEST)
                    extracted["spmeta"] += 1
        inner.close()
        print(f"    本层提取 app 条目 {n} 个")
    outer.close()

    print("提取完成:", extracted)
    # 汇总 pretrained_models 内容
    pm = os.path.join(DEST, "pretrained_models")
    if os.path.isdir(pm):
        for d in sorted(os.listdir(pm)):
            print("pretrained_models/", d)
            sub = os.path.join(pm, d)
            if os.path.isdir(sub):
                for f in sorted(os.listdir(sub))[:40]:
                    print("    -", f)


if __name__ == "__main__":
    main()
