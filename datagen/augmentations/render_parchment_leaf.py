import argparse, random, pathlib, cv2, string
from PIL import Image
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pdf2image import convert_from_path
from weasyprint import HTML
import degradations as dg

TEXT = pathlib.Path("sample_sa.txt").read_text("utf-8").strip()
OUT = pathlib.Path("output_parchment"); OUT.mkdir(exist_ok=True)
DPI = 300
NO_DEG = 0.10

PARCHMENT = ["#f5deb3", "#f0d8ab", "#c3a374", "#a47a3c"]
INKS = ["#100d05", "#23140a"]
DEV = "०१२३४५६७८९"
def _sa(n): return "".join(DEV[int(d)] for d in str(n))

def _env(): return Environment(loader=FileSystemLoader("templates"), autoescape=select_autoescape(["html", "xml"]))

def _morphology():
    op = random.choices(["open", "close", "dilate", "erode"], weights=[1, 1, 1, 2])[0]
    typ = random.choice(["ones", "upper_triangle", "lower_triangle", "x", "plus", "ellipse"])
    sz = 2 if op in ["dilate", "close"] else random.randint(2, 4)
    shp = ((1, sz) if random.choice([True, False]) else (sz, 1)) if typ == "ones" else (sz, sz)
    return dict(operation=op, kernel_type=typ, kernel_shape=shp)

EFFECTS = {
    "blur": (0.2, lambda: dict(radius=random.choice([3, 5]))),
    "bleed_through": (0.15, lambda: dict(alpha=random.uniform(0.7, 0.9), offset_y=random.randint(-8, 8))),
    "salt": (0.2, lambda: dict(amount=random.uniform(0.03, 0.06))),
    "pepper": (0.2, lambda: dict(amount=0.03)),
    "morphology": (0.75, _morphology),
}

def _font(dir_="../fonts"):
    d = list(pathlib.Path(dir_).glob("*.otf")) + list(pathlib.Path(dir_).glob("*.ttf"))
    return str(random.choice(d).resolve())

def _chain():
    c = []
    for n, (p, g) in EFFECTS.items():
        if random.random() < p: c.append((getattr(dg, n), g()))
    return c

def _apply(img, ch):
    for f, kw in ch: img = f(img, **kw)
    return img

def _degrade(p):
    if random.random() < NO_DEG: return Image.open(p)
    img = _apply(cv2.imread(str(p), cv2.IMREAD_COLOR), _chain())
    cv2.imwrite(str(p), img)
    return Image.open(p)

def render(text: str, use_max=False):
    env = _env()
    raw = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = [{"text": x, "large": random.random() < 0.25} for x in raw]

    if use_max:
        margin = 15
        stripe = 0.8
        gap = 0.6
        innerpad = stripe + 4
        line_h = 1.4
        para_spacing = 2.0
        para_scale = 1.4
        font_size = 8
    else:
        margin = random.uniform(8, 15)
        stripe = random.uniform(0.5, 0.8)
        gap = random.uniform(0.2, 0.6)
        innerpad = stripe + random.uniform(2, 4)
        line_h = random.uniform(1.0, random.uniform(1.25, 1.4))
        para_spacing = random.uniform(0.5, 2.0)
        para_scale = random.uniform(1.1, 1.4)
        font_size = random.uniform(5, 8)

    ctx = {
        "font_path": _font(),
        "parchment": random.choice(PARCHMENT),
        "ink": random.choice(INKS),
        "page_width": 180,
        "page_height": 70,
        "font_size": font_size,
        "line_height": line_h,
        "paragraphs": paragraphs,
        "margin_mm": margin,
        "stripe_thick": stripe,
        "inner_left_mm": margin + stripe + gap,
        "inner_right_mm": margin + stripe + gap,
        "text_margin_mm": margin + stripe + gap + stripe,
        "stripe_gap": gap,
        "inner_pad": innerpad,
        "para_spacing": para_spacing,
        "paragraph_scale": para_scale,
        "page_no_sa": _sa(random.randint(1, 999)),
        "show_page_no": random.random() < 0.6,
        "page_side": random.choice(["left", "right"]),
        "page_no_mm": margin / 2,
    }

    pdf = OUT / "tmp.pdf"
    HTML(string=env.get_template("parchment_leaf.html.jinja").render(**ctx), base_url="templates").write_pdf(str(pdf))
    img_path = OUT / "tmp_00.png"
    convert_from_path(str(pdf), dpi=DPI, first_page=1, last_page=1)[0].save(img_path, "PNG")
    pdf.unlink()
    img = _degrade(img_path)
    img_path.unlink()


    lines = []
    if ctx["show_page_no"]: lines.append(ctx["page_no_sa"])
    lines.extend(p["text"] for p in paragraphs)
    caption = "\n".join(lines)
    return img, caption

def _main(k: int, use_max=False):
    for i in range(k):
        img, cap = render(TEXT, use_max=use_max)
        img.save(OUT / f"sample_leaf_{i:03d}.png")
        (OUT / f"sample_leaf_{i:03d}.txt").write_text(cap, "utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max", action="store_true", help="Use max sizing parameters")
    args = ap.parse_args()
    _main(args.n, args.max)
