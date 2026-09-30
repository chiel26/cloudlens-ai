#!/usr/bin/env python3
import os, re, csv, json, math, random, shutil, hashlib, traceback
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import imagehash
import yaml

SEED = 20261001
random.seed(SEED)
TARGET = 1000
MAX_SIDE = 384
JPEG_QUALITY = 88

ROOT = Path.cwd()
WORK = ROOT / ".meklens_work"
CAND = WORK / "candidates"
OUT = WORK / "MekLens_Cloud_Training_1000x10"
BUILD = ROOT / "build"
for p in (WORK, CAND, BUILD):
    p.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "Ac": "Altocumulus",
    "As": "Altostratus",
    "Cb": "Cumulonimbus",
    "Cc": "Cirrocumulus",
    "Ci": "Cirrus",
    "Cs": "Cirrostratus",
    "Cu": "Cumulus",
    "Ns": "Nimbostratus",
    "Sc": "Stratocumulus",
    "St": "Stratus",
}
ALIASES = {
    "Ac": ["ac", "altocumulus", "alto cumulus"],
    "As": ["as", "altostratus", "alto stratus"],
    "Cb": ["cb", "cumulonimbus", "cumulo nimbus"],
    "Cc": ["cc", "cirrocumulus", "cirro cumulus"],
    "Ci": ["ci", "cirrus"],
    "Cs": ["cs", "cirrostratus", "cirro stratus"],
    "Cu": ["cu", "cumulus"],
    "Ns": ["ns", "nimbostratus", "nimbo stratus"],
    "Sc": ["sc", "stratocumulus", "strato cumulus"],
    "St": ["st", "stratus"],
}
VALID_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def norm_text(s):
    s = str(s).lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()

alias_pairs = []
for abbr, aliases in ALIASES.items():
    for a in aliases:
        alias_pairs.append((norm_text(a), abbr))
alias_pairs.sort(key=lambda x: len(x[0]), reverse=True)

def name_to_abbr(name):
    n = norm_text(name)
    for alias, abbr in alias_pairs:
        if n == alias:
            return abbr
    for alias, abbr in alias_pairs:
        if len(alias) >= 4 and alias in n:
            return abbr
    return None

def infer_class(path):
    parts = [norm_text(p) for p in Path(path).parts[:-1]]
    for seg in reversed(parts):
        for alias, abbr in alias_pairs:
            if seg == alias:
                return abbr
    joined = " / ".join(parts)
    for alias, abbr in alias_pairs:
        if len(alias) >= 4 and alias in joined:
            return abbr
    return None

def list_images(root):
    root = Path(root)
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXT]

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def img_phash_int(im):
    return int(str(imagehash.phash(im.convert("RGB"))), 16)

def hamming(a, b):
    return (a ^ b).bit_count()

class HashIndex:
    # 5-band LSH: if Hamming distance <=4, at least one 13-bit band must match.
    def __init__(self):
        self.bands = defaultdict(set)
        self.all = set()
    def _keys(self, h):
        for i in range(5):
            shift = i * 13
            bits = 12 if i == 4 else 13
            yield (i, (h >> shift) & ((1 << bits) - 1))
    def add(self, h):
        if h in self.all:
            return
        self.all.add(h)
        for k in self._keys(h):
            self.bands[k].add(h)
    def near(self, h, maxdist=4):
        if h in self.all:
            return True
        cands = set()
        for k in self._keys(h):
            cands.update(self.bands.get(k, ()))
        return any(hamming(h, x) <= maxdist for x in cands)

old_index = HashIndex()
new_index = HashIndex()
seen_sha = set()
report = {
    "old_sources_status": {},
    "old_images_hashed": 0,
    "candidates_rejected_old_near_duplicate": 0,
    "candidates_rejected_new_near_duplicate": 0,
    "bad_images": 0,
    "source_counts_accepted": {},
}

def normalize_base(im):
    im = ImageOps.exif_transpose(im).convert("RGB")
    if min(im.size) < 96:
        raise ValueError("too_small")
    # Downscale only; preserve aspect ratio.
    im.thumbnail((640, 640), Image.Resampling.LANCZOS)
    return im

def add_candidate(im, abbr, source, license_name, source_url, original_id, is_crop=False):
    if abbr not in CLASSES:
        return False
    try:
        im = normalize_base(im)
        ph = img_phash_int(im)
        if old_index.near(ph, 4):
            report["candidates_rejected_old_near_duplicate"] += 1
            return False
        if new_index.near(ph, 3):
            report["candidates_rejected_new_near_duplicate"] += 1
            return False
        # JPEG bytes are used for exact duplicate detection.
        import io
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=93, optimize=True)
        data = buf.getvalue()
        sh = sha256_bytes(data)
        if sh in seen_sha:
            return False
        seen_sha.add(sh)
        new_index.add(ph)
        folder = CAND / abbr
        folder.mkdir(parents=True, exist_ok=True)
        idx = len(list(folder.glob("*.jpg"))) + 1
        dst = folder / f"{abbr}_base_{idx:05d}.jpg"
        dst.write_bytes(data)
        meta = {
            "class_abbr": abbr,
            "class_en": CLASSES[abbr],
            "candidate_file": str(dst),
            "source_dataset": source,
            "source_license": license_name,
            "source_url": source_url,
            "original_id": str(original_id),
            "phash": f"{ph:016x}",
            "sha256": sh,
            "is_crop": bool(is_crop),
        }
        with open(folder / "meta.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        report["source_counts_accepted"][source] = report["source_counts_accepted"].get(source, 0) + 1
        return True
    except Exception:
        report["bad_images"] += 1
        return False

def build_old_exclusion_hashes():
    try:
        import kagglehub
    except Exception as e:
        report["old_sources_status"]["kagglehub_import"] = repr(e)
        return
    specs = [
        ("CCSN", "mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database"),
        ("Howard-Cloud-X", "imbikramsaha/howard-cloudx"),
    ]
    for source, handle in specs:
        try:
            print(f"[OLD] downloading {source} for duplicate exclusion only...")
            root = Path(kagglehub.dataset_download(handle))
            n = 0
            for p in list_images(root):
                abbr = infer_class(p)
                if abbr not in CLASSES:
                    continue
                try:
                    with Image.open(p) as im:
                        im = normalize_base(im)
                        old_index.add(img_phash_int(im))
                    n += 1
                except Exception:
                    pass
            report["old_sources_status"][source] = {"ok": True, "images_hashed": n}
            report["old_images_hashed"] += n
            print(f"[OLD] {source}: hashed {n:,}")
        except Exception as e:
            report["old_sources_status"][source] = {"ok": False, "error": repr(e)}
            print(f"[OLD] WARNING {source}: {e}")

def load_ccaim():
    source = "CCAiM-CloudsDataset"
    url = "https://huggingface.co/datasets/serbekun/CCAiM-CloudsDataset"
    license_name = "MIT"
    try:
        from datasets import load_dataset
        print("[NEW] loading CCAiM...")
        ds = load_dataset("serbekun/CCAiM-CloudsDataset", split="train")
        feature = ds.features.get("label")
        for i, row in enumerate(ds):
            label = row.get("label")
            try:
                label_name = feature.int2str(label) if hasattr(feature, "int2str") else str(label)
            except Exception:
                label_name = str(label)
            abbr = name_to_abbr(label_name)
            if abbr not in CLASSES:
                continue
            im = row["image"]
            add_candidate(im, abbr, source, license_name, url, f"row:{i}", False)
        print("[NEW] CCAiM done")
    except Exception as e:
        print("[NEW] CCAiM ERROR:", repr(e))
        traceback.print_exc()

def parse_yolo_names(data_yaml):
    d = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8", errors="ignore"))
    names = d.get("names", {})
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}

def find_label_for_image(img_path):
    p = Path(img_path)
    parts = list(p.parts)
    if "images" in parts:
        j = len(parts) - 1 - parts[::-1].index("images")
        lp = Path(*parts[:j], "labels", *parts[j+1:]).with_suffix(".txt")
        if lp.exists():
            return lp
    # fallback common parallel path
    s = str(p).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    lp = Path(s).with_suffix(".txt")
    return lp if lp.exists() else None

def process_yolo_dataset(root, source, license_name, source_url):
    root = Path(root)
    yaml_files = list(root.rglob("data.yaml")) + list(root.rglob("data.yml"))
    if not yaml_files:
        print(f"[RF] {source}: no data.yaml")
        return
    yml = min(yaml_files, key=lambda p: len(p.parts))
    names = parse_yolo_names(yml)
    image_files = list_images(yml.parent)
    accepted = 0
    for p in image_files:
        lp = find_label_for_image(p)
        if not lp:
            continue
        try:
            lines = [x.strip() for x in lp.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            if not lines:
                continue
            with Image.open(p) as original:
                original = ImageOps.exif_transpose(original).convert("RGB")
                W, H = original.size
                for li, line in enumerate(lines):
                    vals = line.split()
                    if len(vals) < 5:
                        continue
                    cid = int(float(vals[0]))
                    abbr = name_to_abbr(names.get(cid, ""))
                    if abbr not in CLASSES:
                        continue
                    x, y, w, h = map(float, vals[1:5])
                    # Slightly expand annotated cloud region.
                    x1 = max(0, (x - w * 0.58) * W)
                    y1 = max(0, (y - h * 0.58) * H)
                    x2 = min(W, (x + w * 0.58) * W)
                    y2 = min(H, (y + h * 0.58) * H)
                    if x2 - x1 < 80 or y2 - y1 < 80:
                        continue
                    crop = original.crop((int(x1), int(y1), int(x2), int(y2)))
                    if add_candidate(crop, abbr, source, license_name, source_url, f"{p.name}#box{li}", True):
                        accepted += 1
        except Exception:
            continue
    print(f"[RF] {source}: accepted {accepted:,} unique crops")

def process_folder_classification(root, source, license_name, source_url):
    accepted = 0
    for p in list_images(root):
        abbr = infer_class(p)
        if abbr not in CLASSES:
            continue
        try:
            with Image.open(p) as im:
                if add_candidate(im.copy(), abbr, source, license_name, source_url, p.name, False):
                    accepted += 1
        except Exception:
            pass
    print(f"[RF] {source}: accepted {accepted:,} classification images")

def download_roboflow_sources():
    try:
        import roboflow
    except Exception as e:
        print("[RF] import error:", e)
        return
    specs = [
        {
            "source": "Roboflow Cloud-2 v3",
            "url": "https://universe.roboflow.com/bakr-alkanbari-hepj4/cloud-2-tqmnt/dataset/3",
            "fmt": "yolov8",
            "license": "CC BY 4.0",
        },
        {
            "source": "Roboflow Cloud-type test v1",
            "url": "https://universe.roboflow.com/mhapongg/cloud-type-test-oiofh/dataset/1",
            "fmt": "folder",
            "license": "CC BY 4.0",
        },
        {
            "source": "Roboflow Cloud Classification 2 v1",
            "url": "https://universe.roboflow.com/clouds-1q8j3/cloud-classification-2-6mni8/dataset/1",
            "fmt": "yolov8",
            "license": "CC BY 4.0",
        },
        {
            "source": "Roboflow Cloud Classification hsharzrf v2",
            "url": "https://universe.roboflow.com/hsharzrf/cloud-classification-mf91q-1hcvv/dataset/2",
            "fmt": "yolov8",
            "license": "CC BY 4.0",
        },
    ]
    for spec in specs:
        try:
            print(f"[RF] downloading {spec['source']}...")
            ds = roboflow.download_dataset(dataset_url=spec["url"], model_format=spec["fmt"])
            loc = Path(ds.location)
            if spec["fmt"] == "folder":
                process_folder_classification(loc, spec["source"], spec["license"], spec["url"].rsplit("/dataset/",1)[0])
            else:
                process_yolo_dataset(loc, spec["source"], spec["license"], spec["url"].rsplit("/dataset/",1)[0])
        except Exception as e:
            print(f"[RF] WARNING {spec['source']}: {repr(e)}")

def square_variant(base, rnd, augment=False):
    im = Image.open(base).convert("RGB")
    w, h = im.size
    if augment:
        # random crop retaining 78-100% of shorter side, no vertical flip.
        scale = rnd.uniform(0.78, 1.0)
        side = int(min(w, h) * scale)
        side = max(96, min(side, w, h))
        x0 = 0 if w == side else rnd.randint(0, w - side)
        y0 = 0 if h == side else rnd.randint(0, h - side)
        im = im.crop((x0, y0, x0 + side, y0 + side))
        if rnd.random() < 0.5:
            im = ImageOps.mirror(im)
        angle = rnd.uniform(-3.0, 3.0)
        im = im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        im = ImageEnhance.Brightness(im).enhance(rnd.uniform(0.82, 1.18))
        im = ImageEnhance.Contrast(im).enhance(rnd.uniform(0.82, 1.18))
        im = ImageEnhance.Color(im).enhance(rnd.uniform(0.88, 1.12))
        if rnd.random() < 0.25:
            im = im.filter(ImageFilter.GaussianBlur(rnd.uniform(0.15, 0.7)))
    else:
        side = min(w, h)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        im = im.crop((x0, y0, x0 + side, y0 + side))
    im = im.resize((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return im

def read_meta(abbr):
    p = CAND / abbr / "meta.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows

def build_output():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = []
    counts = []
    for abbr, class_en in CLASSES.items():
        metas = read_meta(abbr)
        if not metas:
            raise RuntimeError(f"No base images available for {class_en}")
        rnd = random.Random(SEED + sum(map(ord, abbr)))
        rnd.shuffle(metas)
        folder = OUT / class_en
        folder.mkdir(parents=True, exist_ok=True)
        base_n = min(TARGET, len(metas))
        selected = metas[:base_n]
        out_i = 0
        out_sha = set()
        # unique originals first
        for meta in selected:
            out_i += 1
            im = square_variant(meta["candidate_file"], rnd, augment=False)
            dst = folder / f"{abbr}_{out_i:04d}.jpg"
            im.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
            sh = hashlib.sha256(dst.read_bytes()).hexdigest()
            out_sha.add(sh)
            manifest.append({
                "class_abbr": abbr, "class_en": class_en, "output_file": str(dst.relative_to(OUT)),
                "source_dataset": meta["source_dataset"], "source_license": meta["source_license"],
                "source_url": meta["source_url"], "original_id": meta["original_id"],
                "is_augmented": False, "base_phash": meta["phash"], "output_sha256": sh
            })
        # fill to TARGET using deterministic augmentations of new, non-old base images
        attempts = 0
        while out_i < TARGET:
            attempts += 1
            if attempts > TARGET * 100:
                raise RuntimeError(f"Could not fill {class_en}")
            meta = metas[(out_i + attempts) % len(metas)]
            local_rnd = random.Random(SEED * 1000003 + out_i * 9176 + attempts * 37 + sum(map(ord, abbr)))
            im = square_variant(meta["candidate_file"], local_rnd, augment=True)
            import io
            b = io.BytesIO()
            im.save(b, "JPEG", quality=JPEG_QUALITY, optimize=True)
            data = b.getvalue()
            sh = hashlib.sha256(data).hexdigest()
            if sh in out_sha:
                continue
            out_i += 1
            out_sha.add(sh)
            dst = folder / f"{abbr}_{out_i:04d}.jpg"
            dst.write_bytes(data)
            manifest.append({
                "class_abbr": abbr, "class_en": class_en, "output_file": str(dst.relative_to(OUT)),
                "source_dataset": meta["source_dataset"], "source_license": meta["source_license"],
                "source_url": meta["source_url"], "original_id": meta["original_id"],
                "is_augmented": True, "base_phash": meta["phash"], "output_sha256": sh
            })
        counts.append({
            "class_abbr": abbr, "class_en": class_en, "total": TARGET,
            "distinct_new_base_images": len(metas),
            "originals_used_without_augmentation": base_n,
            "augmented_fill": TARGET - base_n,
        })
        print(f"[OUT] {class_en}: base unique={len(metas)}, output={TARGET}, augmented={TARGET-base_n}")
    # manifests
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=manifest[0].keys())
        w.writeheader(); w.writerows(manifest)
    with open(OUT / "class_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=counts[0].keys())
        w.writeheader(); w.writerows(counts)
    (OUT / "dedupe_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    license_text = """# MekLens AI training dataset — licensing and attribution

This package contains 10 WMO cloud-genera folders, balanced at 1,000 images per class.

Sources used for NEW candidate imagery:
1. CCAiM-CloudsDataset — MIT License
   https://huggingface.co/datasets/serbekun/CCAiM-CloudsDataset
2. Cloud-2 — CC BY 4.0
   https://universe.roboflow.com/bakr-alkanbari-hepj4/cloud-2-tqmnt
3. Cloud-type test — CC BY 4.0
   https://universe.roboflow.com/mhapongg/cloud-type-test-oiofh
4. Cloud Classification 2 — CC BY 4.0
   https://universe.roboflow.com/clouds-1q8j3/cloud-classification-2-6mni8
5. Cloud Classification (hsharzrf) — CC BY 4.0
   https://universe.roboflow.com/hsharzrf/cloud-classification-mf91q-1hcvv

CC BY 4.0 requires attribution. Keep this file with redistributed copies and cite the
source project(s) listed in manifest.csv. Crops and augmentations are marked in manifest.csv.

Previous MekLens/CloudLens sources CCSN and Howard-Cloud-X are used ONLY as a
duplicate-exclusion reference when available during the build and are not redistributed
from those old packages by this build.
"""
    (OUT / "LICENSES_AND_ATTRIBUTION.md").write_text(license_text, encoding="utf-8")
    readme = f"""# MekLens AI — Additional Cloud Training Images

- Classes: 10
- Images per class: {TARGET}
- Total images: {TARGET * len(CLASSES)}
- Output size: {MAX_SIDE}x{MAX_SIDE} JPEG
- Exact output duplicate protection: SHA-256
- New-source near-duplicate filtering: perceptual hash
- Old-source exclusion: perceptual hashes from CCSN + Howard-Cloud-X when public downloads succeeded
- Full provenance: manifest.csv
- Per-class counts: class_counts.csv
- Duplicate audit: dedupe_report.json
- License details: LICENSES_AND_ATTRIBUTION.md

Folders are named for direct upload to Teachable Machine.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    return counts

def main():
    print("=== MekLens AI dataset builder ===")
    build_old_exclusion_hashes()
    load_ccaim()
    download_roboflow_sources()
    counts = build_output()
    BUILD.mkdir(parents=True, exist_ok=True)
    zip_base = BUILD / "MekLens_Cloud_Training_1000x10"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=OUT)
    size_mb = Path(zip_path).stat().st_size / 1024 / 1024
    print("=== DONE ===")
    print("ZIP:", zip_path)
    print("ZIP MB:", round(size_mb, 2))
    print(json.dumps(counts, indent=2))

if __name__ == "__main__":
    main()
