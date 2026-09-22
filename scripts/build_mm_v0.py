"""mm-v0: the first multimodal decision set. Public image and audio datasets turned into labelled decision records, each
with its media extracted to a file next to the JSONL and referenced by path. Every source is a decision text alone
cannot make: the answer is in the picture or in the voice.

    aokvqa       photo + a multiple-choice question about it (A-OKVQA; train -> train, validation -> test)
    pets         photo: cat or dog, and which breed among six (Oxford-IIIT Pet; train -> train, test -> test)
    screenshot   a customer message rendered as an image (chat, email, note, dark mode): which of 77 banking intents
                 (Banking77 text; train -> train, test -> test). The model has to read it.
    minds14      a real recorded banking call: which of 14 intents, and a yes/no on one intent (MInDS-14 en-US/GB/AU; no
                 test split, so 20% by a hash of the file name)
    crema        an actor saying one of 12 neutral sentences in an emotion: which emotion, and a yes/no on one (CREMA-D; the
                 words carry no emotion, only the voice does; clips where the acted and the crowd-perceived emotion agree;
                 actors 1076-1091 held out for test)

Run: uv run --extra media python scripts/build_mm_v0.py --out data/mm-v0 [--n_train 1000 --n_test 300]
Writes <out>/{train,test}.jsonl (kev.data.load_records format plus "media") and <out>/media/<source>/...
"""
import argparse, hashlib, io, json, random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPOS = {"aokvqa": "HuggingFaceM4/A-OKVQA", "pets": "timm/oxford-iiit-pet", "banking77": "legacy-datasets/banking77",
         "minds14": "PolyAI/minds14", "crema": "AbstractTTS/CREMA-D"}
MINDS14 = {"abroad": "Using the card or account abroad", "address": "Changing the address on file", "app_error": "A problem with the banking app",
           "atm_limit": "The ATM withdrawal limit", "balance": "Checking the account balance", "business_loan": "A loan for a business",
           "card_issues": "A problem with a card", "cash_deposit": "Depositing cash", "direct_debit": "Setting up or changing a direct debit",
           "freeze": "Freezing the card or account", "high_value_payment": "Making a large payment", "joint_account": "Opening or using a joint account",
           "latest_transactions": "Recent transactions on the account", "pay_bill": "Paying a bill"}
CREMA = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
CREMA_PERCEIVED = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
FONTS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Georgia.ttf",
         "/System/Library/Fonts/Supplemental/Courier New.ttf", "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Supplemental/Verdana.ttf"]


def seed_of(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def files(repo, prefix):
    api = HfApi(); sha = api.dataset_info(repo).sha
    names = sorted(f for f in api.list_repo_files(repo, repo_type="dataset", revision=sha) if f.startswith(prefix) and f.endswith(".parquet"))
    return sha, [hf_hub_download(repo, f, repo_type="dataset", revision=sha) for f in names]


def rows(repo, prefix, columns=None):
    sha, paths = files(repo, prefix)
    out = []
    for p in paths: out += pq.read_table(p, columns=columns).to_pylist()
    return sha, out


def save(out, source, name, raw):
    path = out / "media" / source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return str(path.relative_to(out))


def shrink_jpeg(raw, side=768):
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB"); im.thumbnail((side, side))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); return buf.getvalue()


def meta(source, split, row, repo, sha, key):
    return {"source": source, "repo": repo, "revision": sha, "split": split, "row": row, "id": f"{source}/{split}/{key}", "group_id": f"{source}/{split}/{key}"}


def instr(text, rng):
    return {"question": text, "focus": rng.choice(["Use only what the attachment shows.", "Pick the single best fit.", "Look at the whole attachment."])} if rng.random() < 0.15 else text


# --- images -----------------------------------------------------------------------------------------------------------

def aokvqa(out, split, n, rng):
    sha, rs = rows(REPOS["aokvqa"], f"data/{'train' if split == 'train' else 'validation'}-")
    recs = []
    for i in rng.sample(range(len(rs)), min(n, len(rs))):
        ex = rs[i]; order = list(range(len(ex["choices"]))); rng.shuffle(order)
        crit = {f"opt_{k + 1}": ex["choices"][j] for k, j in enumerate(order)}
        label = f"opt_{order.index(ex['correct_choice_idx']) + 1}"
        path = save(out, "aokvqa", f"{split}/{ex['question_id']}.jpg", shrink_jpeg(ex["image"]["bytes"]))
        recs.append({"state": "", "media": [{"type": "image", "path": path}], "questions": {
            "answer": {"type": "choice", "instructions": instr(ex["question"], rng), "criteria": crit, "label": label, "src": "aokvqa"}},
            "_meta": meta("aokvqa", split, i, REPOS["aokvqa"], sha, ex["question_id"])})
    return recs


def pets(out, split, n, rng):
    sha, rs = rows(REPOS["pets"], f"data/{split}-")
    import pyarrow.parquet as _pq
    names = json.loads(_pq.read_schema(files(REPOS["pets"], f"data/{split}-")[1][0]).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    species = {}
    for ex in rs: species[ex["label"]] = ex["label_cat_dog"]
    recs = []
    for i in rng.sample(range(len(rs)), min(n, len(rs))):
        ex = rs[i]; y = ex["label"]; animal = "dog" if ex["label_cat_dog"] == 1 else "cat"
        pool = [k for k in species if k != y and (species[k] == species[y] or rng.random() < 0.3)]
        opts = rng.sample(pool, 5) + [y]; rng.shuffle(opts)
        crit = {names[k]: (names[k].replace("_", " ").title() if rng.random() < 0.5 else None) for k in opts}
        ask = rng.choice(["cat", "dog"])
        path = save(out, "pets", f"{split}/{ex['image_id']}.jpg", shrink_jpeg(ex["image"]["bytes"]))
        recs.append({"state": "", "media": [{"type": "image", "path": path}], "questions": {
            "breed": {"type": "choice", "instructions": instr("Which breed is the animal in the photo?", rng), "criteria": crit, "label": names[y], "src": "pets_breed"},
            f"is_{ask}": {"type": "noul", "instructions": f"Is the animal in the photo a {ask}?", "label": animal == ask, "src": "pets_species"}},
            "_meta": meta("pets", split, i, REPOS["pets"], sha, ex["image_id"])})
    return recs


def render_message(text, rng):
    """A customer message as a screenshot: chat bubble, email, sticky note or dark-mode chat, in a random system font."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap
    style = rng.choice(["chat", "email", "note", "dark"])
    fonts = [f for f in FONTS if Path(f).exists()]
    size = rng.randint(18, 28)
    font = ImageFont.truetype(rng.choice(fonts), size) if fonts else ImageFont.load_default(size)
    width = rng.randint(28, 46)
    body = textwrap.fill(text, width)
    if style == "email":
        name = rng.choice(["Alex", "Sam", "Jordan", "Priya", "Chen", "Maria", "Tom", "Aisha"])
        body = f"From: {name.lower()}@example.com\nTo: support@bank.example\nSubject: {rng.choice(['Help', 'Question', 'Issue with my account', 'Urgent', 'Hi'])}\n\n{body}\n\nThanks,\n{name}"
    bg, fg, bubble = {"chat": ("#f2f2f7", "#111111", "#ffffff"), "email": ("#ffffff", "#202124", "#ffffff"),
                      "note": ("#fff7b0", "#3a3a00", "#fff7b0"), "dark": ("#1c1c1e", "#f2f2f7", "#2c2c2e")}[style]
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    x0, y0, x1, y1 = probe.multiline_textbbox((0, 0), body, font=font, spacing=6)
    pad = rng.randint(16, 40)
    im = Image.new("RGB", (x1 - x0 + 4 * pad, y1 - y0 + 4 * pad), bg)
    d = ImageDraw.Draw(im)
    if style in ("chat", "dark"): d.rounded_rectangle((pad, pad, x1 - x0 + 3 * pad, y1 - y0 + 3 * pad), radius=18, fill=bubble)
    d.multiline_text((2 * pad, 2 * pad), body, font=font, fill=fg, spacing=6)
    buf = io.BytesIO(); im.save(buf, "PNG"); return buf.getvalue()


def screenshot(out, split, n, rng):
    sha, rs = rows(REPOS["banking77"], f"data/{split}-")
    names = json.loads(pq.read_schema(files(REPOS["banking77"], f"data/{split}-")[1][0]).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    tmpl = rng.choice
    recs = []
    for i in rng.sample(range(len(rs)), min(n, len(rs))):
        ex = rs[i]
        t = tmpl(["Customer asks about {}", "Issue concerning {}", "Request related to {}", "{}"])
        crit = {k: (t.format(k.replace("_", " ")) if rng.random() < 0.5 else None) for k in names}
        path = save(out, "screenshot", f"{split}/{i}.png", render_message(ex["text"], rng))
        recs.append({"state": rng.choice(["", "Screenshot attached by the customer.", {"channel": rng.choice(["chat", "email", "web form"]), "attachment": "screenshot"}]),
                     "media": [{"type": "image", "path": path}], "questions": {
            "intent": {"type": "choice", "instructions": instr("Which banking intent best describes the customer's message in the screenshot?", rng),
                       "criteria": crit, "label": names[ex["label"]], "src": "screenshot"}},
            "_meta": meta("screenshot", split, i, REPOS["banking77"], sha, i)})
    return recs


# --- audio ------------------------------------------------------------------------------------------------------------

def minds14(out, split, n, rng):
    recs, keys = [], list(MINDS14)
    for loc in ("en-US", "en-GB", "en-AU"):
        sha, rs = rows(REPOS["minds14"], f"{loc}/")
        for i, ex in enumerate(rs):
            held = int(hashlib.sha256(ex["path"].encode()).hexdigest(), 16) % 5 == 0
            if held != (split == "test"): continue
            y = keys[ex["intent_class"]]; ask = rng.choice([y, rng.choice(keys)])
            path = save(out, "minds14", ex["path"].replace("~", "/"), ex["audio"]["bytes"])
            recs.append({"state": rng.choice(["", "Voicemail left on the support line.", {"channel": "phone", "attachment": "call recording"}]),
                         "media": [{"type": "audio", "path": path}], "questions": {
                "intent": {"type": "choice", "instructions": instr("What does the caller want?", rng),
                           "criteria": {k: (v if rng.random() < 0.7 else None) for k, v in MINDS14.items()}, "label": y, "src": "minds14"},
                f"is_{ask}": {"type": "noul", "instructions": f"Is the caller asking about: {MINDS14[ask].lower()}?", "label": ask == y, "src": "minds14_yn"}},
                "_meta": meta("minds14", split, i, REPOS["minds14"], sha, ex["path"])})
    rng.shuffle(recs)
    return recs[:n]


def crema(out, split, n, rng):
    sha, rs = rows(REPOS["crema"], "data/", columns=["file", "audio", "major_emotion"])
    keys = list(CREMA.values())
    pool = []
    for i, ex in enumerate(rs):
        actor, _, code, _ = ex["file"].removesuffix(".wav").split("_")
        if (int(actor) >= 1076) != (split == "test") or ex["major_emotion"] != CREMA_PERCEIVED[code]: continue
        pool.append((i, ex, CREMA[code]))
    recs = []
    for i, ex, y in rng.sample(pool, min(n, len(pool))):
        ask = rng.choice([y, rng.choice(keys)])
        path = save(out, "crema", ex["file"], ex["audio"]["bytes"])
        recs.append({"state": rng.choice(["", "Voice message from a customer.", {"channel": "phone"}]), "media": [{"type": "audio", "path": path}], "questions": {
            "emotion": {"type": "choice", "instructions": instr("How does the speaker sound?", rng), "criteria": {k: None for k in keys}, "label": y, "src": "crema"},
            f"sounds_{ask}": {"type": "noul", "instructions": f"Does the speaker sound {ask}?", "label": ask == y, "src": "crema_yn"}},
            "_meta": meta("crema", split, i, REPOS["crema"], sha, ex["file"])})
    return recs


SOURCES = {"aokvqa": aokvqa, "pets": pets, "screenshot": screenshot, "minds14": minds14, "crema": crema}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mm-v0"); ap.add_argument("--n_train", type=int, default=1000); ap.add_argument("--n_test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", a.n_train), ("test", a.n_test)):
        recs = []
        for name, fn in SOURCES.items():
            if a.only and name not in a.only.split(","): continue
            got = fn(out, split, n, random.Random(seed_of(a.seed, name, split)))
            print(f"{split} {name}: {len(got)}", flush=True); recs += got
        random.Random(seed_of(a.seed, split)).shuffle(recs)
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}/train.jsonl, {out}/test.jsonl")


if __name__ == "__main__":
    main()
