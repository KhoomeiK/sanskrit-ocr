import argparse, random, pathlib, cv2, numpy as np, imageio.v2 as imageio, string
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
from pdf2image import convert_from_path
import degradations as dg
from tqdm import tqdm

# ─── constants ─────────────────────────────────────────────
TEXT    = pathlib.Path("sample_sa.txt").read_text("utf-8").strip()
OUT     = pathlib.Path("output_parchment")
OUT.mkdir(exist_ok=True)
DPI     = 300
NO_DEG  = 0.10

PARCHMENT = ["#f5deb3", "#f0d8ab", "#c3a374", "#a47a3c"]
INKS      = ["#100d05", "#23140a"]
DEV       = "०१२३४५६७८९"
def _sa_num(n):
    return "".join(DEV[int(d)] for d in str(n))


# ─── helper for morphology effect ──────────────────────────
def _morphology_params():
    ops     = ["open", "close", "dilate", "erode"]
    weights = [1, 1, 1, 2]  # erode twice as likely
    operation = random.choices(ops, weights=weights, k=1)[0]
    kernel_type = random.choice([
        "ones", "upper_triangle", "lower_triangle",
        "x", "plus", "ellipse"
    ])
    if operation in ["dilate", "close"]:
        size = 2
    else:
        size = random.randint(2, 4)
    if kernel_type == "ones":
        kernel_shape = random.choice([(1, size), (size, 1)])
    else:
        kernel_shape = (size, size)
    return {
        "operation":     operation,
        "kernel_type":   kernel_type,
        "kernel_shape":  kernel_shape
    }


# ─── Genalog-style effects ─────────────────────────────────
EFFECTS = {
    "blur":          (0.3,  lambda: dict(radius=random.choice([3, 5, 7]))),
    "bleed_through": (0.15, lambda: dict(
                          alpha=random.uniform(0.7, 0.9),
                          offset_y=random.randint(-8, 8)
                      )),
    "salt":          (0.2,  lambda: dict(amount=random.uniform(0.03, 0.06))),
    "pepper":        (0.2,  lambda: dict(amount=0.03)),
    "morphology":    (0.75, _morphology_params),
}


def _choose_font(dir_="../fonts"):
    files = list(pathlib.Path(dir_).glob("*.otf")) + list(pathlib.Path(dir_).glob("*.ttf"))
    return str(random.choice(files).resolve())


def _env():
    return Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["html", "xml"])
    )


def _random_chain():
    chain = []
    for name, (prob, gen) in EFFECTS.items():
        if random.random() < prob:
            chain.append((getattr(dg, name), gen()))
    return chain


def _apply_chain(img, chain):
    for fn, kw in chain:
        img = fn(img, **kw)
    return img


def _degrade(p):
    """
    Degrade the image at Path p and return (output_path, chain_applied).
    If no degradation or NO_DEG hit, returns (p, []).
    """
    if random.random() < NO_DEG:
        return p, []

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    chain = _random_chain()
    if not chain:
        return p, []

    out = _apply_chain(img, chain)
    deg = p.with_name(p.stem + "-deg.png")
    cv2.imwrite(str(deg), out)
    p.unlink()
    return deg, chain


def _pdf2png(pdf, stem):
    pages = []
    for i, pg in enumerate(convert_from_path(pdf, dpi=DPI)):
        fn = f"{stem}_{i:02d}.png"
        pg.save(fn, "PNG")
        pages.append(pathlib.Path(fn))
    return pages


def generate(n):
    env = _env()
    frames = []
    raw_paras = [p.strip() for p in TEXT.split("\n\n") if p.strip()]

    # Open degradation log
    log_path = OUT / "degradation_log.txt"
    with log_path.open("w", encoding="utf-8") as log_f:
        for i in tqdm(range(n)):
            margin      = round(random.uniform(8, 15), 2)
            stripe      = round(random.uniform(0.5, 0.8), 2)
            gap         = round(random.uniform(0.2, 0.6), 2)
            innerpad    = round(stripe + random.uniform(2, 4), 2)
            max_lh      = random.uniform(1.25, 1.4)
            line_h      = random.uniform(1.0, max_lh)
            para_spc    = round(random.uniform(0.5, 2.0), 2)
            para_scale  = round(random.uniform(1.1, 1.4), 2)

            inner_left   = margin + stripe + gap
            safe_text_mm = margin + stripe + gap + stripe
            page_no_mm   = margin / 2

            paragraphs = []
            for ptext in raw_paras:
                paragraphs.append({
                    "text": ptext,
                    "large": random.random() < 0.25
                })

            ctx = {
                "font_path":       _choose_font(),
                "parchment":       random.choice(PARCHMENT),
                "ink":             random.choice(INKS),

                "page_width":      180,
                "page_height":     70,
                "font_size":       random.uniform(6, 9),
                "line_height":     line_h,

                "paragraphs":      paragraphs,
                "margin_mm":       margin,
                "stripe_thick":    stripe,
                "inner_left_mm":   inner_left,
                "inner_right_mm":  inner_left,
                "text_margin_mm":  safe_text_mm,
                "stripe_gap":      gap,
                "inner_pad":       innerpad,

                "para_spacing":    para_spc,
                "paragraph_scale": para_scale,

                "page_no_sa":      _sa_num(random.randint(1, 999)),
                "show_page_no":    random.random() < 0.6,
                "page_side":       random.choice(["left", "right"]),
                "page_no_mm":      page_no_mm,
            }

            html = env.get_template("parchment_leaf.html.jinja").render(**ctx)
            base = OUT / f"leaf_{i:03d}"
            pdf  = str(base) + ".pdf"
            HTML(string=html, base_url="templates").write_pdf(pdf)

            for page_idx, p in enumerate(_pdf2png(pdf, str(base))):
                out_path, chain = _degrade(p)
                frames.append(imageio.imread(str(out_path)))

                # Build human-readable description
                if chain:
                    ops = []
                    for fn, kw in chain:
                        name = fn.__name__
                        params = ",".join(f"{k}={v}" for k, v in kw.items())
                        ops.append(f"{name}({params})")
                    desc = "; ".join(ops)
                else:
                    desc = "no degradation"

                log_f.write(f"{out_path.name}: {desc}\n")

            pathlib.Path(pdf).unlink()

    # build GIF
    mxh = max(f.shape[0] for f in frames)
    mxw = max(f.shape[1] for f in frames)
    norm = []
    for f in frames:
        if f.ndim == 2:
            f = np.stack([f] * 3, -1)
        if f.shape[2] == 4:
            f = f[..., :3]
        pad = ((0, mxh - f.shape[0]), (0, mxw - f.shape[1]), (0, 0))
        norm.append(np.pad(f, pad, constant_values=255))
    imageio.mimsave(str(OUT / "showcase.gif"), norm, duration=250)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=10)
    generate(ap.parse_args().num)