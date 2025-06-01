import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from augmentations import degradations as dg
import argparse, cv2, pathlib, random, string
from PIL import Image
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pdf2image import convert_from_path
from weasyprint import HTML

BASE_DIR = Path(__file__).parent.parent.parent
TEXT_FILE = BASE_DIR / "sample_sa.txt"
TEXT = TEXT_FILE.read_text(encoding="utf-8").strip() if TEXT_FILE.exists() else ""
OUT = pathlib.Path("output"); OUT.mkdir(exist_ok=True)
NO_DEGRADE = 0.1
DPI = 300

LAYOUTS = ["vanilla", "columns", "footnote", "subheading"]
WEIGHTS = [0.25, 0.25, 0.25, 0.25]

s = 2

EFFECTS = {
    "blur": (0.3, lambda: dict(radius=random.choice([3, 5, 7]))),
    "bleed_through": (0.15, lambda: dict(alpha=random.uniform(0.7, 0.9), offset_y=random.randint(-10, 10))),
    "salt": (0.2, lambda: dict(amount=random.uniform(0.01, 0.05))),
    "pepper": (0.2, lambda: dict(amount=0.03)),
    "morphology": (0.75, lambda: (
        lambda k: dict(
            operation=random.choices(["open", "close", "dilate", "erode"], weights=[1, 1, 1, 2])[0],
            kernel_type=k,
            kernel_shape=((1, s) if random.choice([True, False]) else (s, 1)) if k == "ones" else (s, s)
        )
    )(k := random.choice(["ones", "upper_triangle", "lower_triangle", "x", "plus", "ellipse"]))),
}

DEV = "०१२३४५६७८९"
def _dev(n): return "".join(DEV[int(d)] for d in str(n))

def _rand_phrase(t, a=1, b=3, m=3):
    w = [x.strip(string.punctuation + "“”’‘\"") for x in t.split()]
    w = [x for x in w if len(x) >= m]
    n = min(random.randint(a, b), len(w))
    return " ".join(random.sample(w, n)) if w else "अध्यायः"

def _rand_footnotes(t): return [_rand_phrase(t, 4, 8) for _ in range(random.randint(1, 2))]

def _font(): 
    fonts_dir = BASE_DIR / "fonts"
    return random.choice(list(fonts_dir.glob("*.?tf"))).resolve()

def _env(): 
    templates_dir = BASE_DIR / "augmentations" / "templates"
    return Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=select_autoescape(["html", "xml"]))

def _build_html(env, ctx):
    html = env.get_template("book_page.html.jinja").render(**ctx)
    reset = f'<style>@page{{counter-reset: page {ctx["page_start"]};}}</style>'
    return html.replace("</head>", reset + "</head>")

def _chain():
    ch = []
    for n, (p, g) in EFFECTS.items():
        if random.random() < p: ch.append((getattr(dg, n), g()))
    return ch

def _apply(img, ch):
    for f, kw in ch: img = f(img, **kw)
    return img

def _degrade(p):
    if random.random() < NO_DEGRADE: return Image.open(p)
    img = _apply(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), _chain())
    cv2.imwrite(str(p), img)
    return Image.open(p)

def render(text: str, use_max=False):
    env = _env()
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    mode = random.choices(LAYOUTS, weights=WEIGHTS, k=1)[0]

    # choose a font and decide base font_size
    font_path = str(_font())

    if use_max:
        column_gap = 10
        column_rule_width = 0.6
        margin_mm = 20
        footnote_size = 9
        footnote_rule = 1.2
        font_size = 12
        line_height = 1.5
        header_size = 12
        subhead_size = 11
        subhead_margin = 10
    else:
        column_gap = random.randint(6, 10)
        column_rule_width = random.uniform(0.3, 0.6)
        margin_mm = random.randint(12, 20)
        footnote_size = random.uniform(8, 9)
        footnote_rule = random.uniform(0.8, 1.2)
        font_size = random.uniform(8, 12)
        line_height = random.uniform(1.25, 1.5)
        header_size = random.uniform(10, 12)
        subhead_size = random.uniform(9, 11)
        subhead_margin = random.randint(6, 10)

    # if the font filename contains "Sharad", force the smallest size (8)
    if "Sharad" in Path(font_path).name:
        font_size = 8

    ctx = dict(
        font_path=font_path,
        chapter_title=_rand_phrase(text, 2, 3),
        page_number_pos=random.choice(["top", "bottom"]),
        page_side=random.choice(["left", "right"]),
        mode=mode,
        paragraphs=paras,
        column_gap=column_gap,
        column_rule_width=column_rule_width,
        subheading=_rand_phrase(text, 2, 3),
        subhead_at=random.randint(1, max(1, len(paras)//2)),
        footnotes=_rand_footnotes(text),
        margin_mm=margin_mm,
        footnote_size=footnote_size,
        footnote_rule=footnote_rule,
        font_size=font_size,
        line_height=line_height,
        header_size=header_size,
        weight_title=random.choice([500, 700]),
        weight_body=random.choice([300, 700]),
        weight_pageno=random.choice([400, 500]),
        page_width=180,
        page_height=180,
        page_start=random.randint(1, 999),
        subhead_size=subhead_size,
        subhead_margin=subhead_margin,
    )

    pdf = OUT / "tmp.pdf"
    HTML(string=_build_html(env, ctx), base_url=str(BASE_DIR / "augmentations" / "templates")).write_pdf(str(pdf))
    img_path = OUT / "tmp_00.png"
    convert_from_path(str(pdf), dpi=DPI, first_page=1, last_page=1)[0].save(img_path, "PNG")
    pdf.unlink()
    img = _degrade(img_path)
    img_path.unlink()
    page = _dev(ctx["page_start"])

    header = ctx["chapter_title"]
    if ctx["page_number_pos"] == "top":
        header = f"{page} {header}" if ctx["page_side"] == "left" else f"{header} {page}"

    lines = [header]
    for i, p in enumerate(paras):
        if mode == "subheading" and i == ctx["subhead_at"]: lines.append(ctx["subheading"])
        lines.append(p)
    if mode == "footnote":
        lines.append("")
        lines.extend(ctx["footnotes"])
    if ctx["page_number_pos"] == "bottom": lines.append(page)

    caption = "\n".join(lines)
    return img, caption

def _main(k: int, use_max=False):
    for i in range(k):
        img, cap = render(TEXT, use_max=use_max)
        img.save(OUT / f"sample_book_{i:03d}.png")
        (OUT / f"sample_book_{i:03d}.txt").write_text(cap, "utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max", action="store_true", help="Use max sizing parameters")
    args = ap.parse_args()
    _main(args.n, args.max)

