#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, gzip, re, time, urllib.request, shutil, random
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
RAW = DATA / "raw"
RAW.mkdir(parents=True, exist_ok=True)

CAT_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
CAT_PATH = RAW / "pg_catalog.csv.gz"
SEP = "<|endoftext|>"

UA = "Academic-Project (mailto:you@example.com)"
def dl(url, path):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f)

def candidate_urls(gid:int):
    base=f"https://www.gutenberg.org/cache/epub/{gid}/"
    for n in [f"pg{gid}.txt.utf8", f"pg{gid}.txt", f"{gid}.txt.utf8", f"{gid}.txt", f"{gid}-0.txt"]:
        yield base+n

def fetch_text(gid:int, timeout=30):
    for url in candidate_urls(gid):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw=r.read()
                try: return raw.decode("utf-8")
                except UnicodeDecodeError: return raw.decode("latin-1","ignore")
        except Exception:
            pass
    return None

def strip_boiler(t:str)->str:
    s = re.compile(r"^\s*\*\*\*\s*START OF [\s\S]*?\*\*\*\s*$", re.MULTILINE)
    e = re.compile(r"^\s*\*\*\*\s*END OF [\s\S]*?\*\*\*\s*$", re.MULTILINE)
    S, E = list(s.finditer(t)), list(e.finditer(t))
    if S and E:
        t = t[S[-1].end():E[0].start()]
    t = re.sub(r"\r\n?", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def main(subset_size=100, min_len=2000, sleep=0.6, seed=42):
    random.seed(seed)
    DATA.mkdir(parents=True, exist_ok=True)
    if not CAT_PATH.exists():
        print("baixando catálogo…")
        dl(CAT_URL, CAT_PATH)

    pt_ids = []
    with gzip.open(CAT_PATH, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            langs = {x.strip().lower() for x in (row.get("Language") or "").replace(",", ";").split(";") if x.strip()}
            tid = (row.get("Text#") or row.get("Text_Number") or row.get("TextNum") or "").strip()
            if "pt" in langs and tid.isdigit():
                pt_ids.append(int(tid))

    pt_ids = sorted(set(pt_ids))
    print(f"IDs PT no catálogo: {len(pt_ids)}")

    texts = []
    kept = 0
    for i, gid in enumerate(pt_ids[:subset_size*2], 1):
        if len(texts) >= subset_size:
            break
        txt = fetch_text(gid)
        time.sleep(sleep)
        if not txt:
            continue
        body = strip_boiler(txt)
        if len(body) < min_len:
            continue
        texts.append(body)
        kept += 1
        if kept % 10 == 0:
            print(f"[coletados úteis] {kept}")

    if not texts:
        raise SystemExit("Nenhum texto PT útil encontrado; tente aumentar subset_size ou relaxar min_len.")

    corpus = f" {SEP} ".join(texts)
    n = len(corpus)
    train_txt, val_txt, test_txt = corpus[:int(0.8*n)], corpus[int(0.8*n):int(0.9*n)], corpus[int(0.9*n):]

    (DATA/'train.txt').write_text(train_txt, encoding="utf-8")
    (DATA/'val.txt').write_text(val_txt, encoding="utf-8")
    (DATA/'test.txt').write_text(test_txt, encoding="utf-8")
    print("salvos:", DATA/'train.txt', DATA/'val.txt', DATA/'test.txt')

if __name__ == "__main__":
    main()
