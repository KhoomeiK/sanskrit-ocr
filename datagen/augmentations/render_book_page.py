import degradations as dg

import argparse, cv2, pathlib, random, string
import imageio.v2 as imageio
from jinja2 import Environment, FileSystemLoader, select_autoescape
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm
from weasyprint import HTML


TEXT = open("sample_sa.txt", encoding="utf-8").read().strip()
OUTPUT_DIR = pathlib.Path("output"); OUTPUT_DIR.mkdir(exist_ok=True)
NO_DEGRADE_PROB = 0.1
IMAGE_DPI = 300

LAYOUTS = ["vanilla","columns","footnote","subheading"]
WEIGHTS = [0.25,      0.25,       0.25,      0.25]

EFFECTS = {
    "blur":             (0.3, lambda: dict(radius=random.choice([3,5,7]))),
    "bleed_through":    (0.15, lambda: dict(alpha=round(random.uniform(0.7,0.9),2),
                                           offset_y=random.randint(-10,10))),
    "salt":             (0.2, lambda: dict(amount=round(random.uniform(0.01,0.05),3))),
    "pepper":           (0.2, lambda: dict(amount=0.03)),
    # "salt_then_pepper": (0.25, lambda: dict(salt_amount=0.05, pepper_amount=0.03)),
    "morphology":       (0.75, lambda: (
                            lambda kernel_type: dict(
                                operation=random.choices(
                                    ["open", "close", "dilate", "erode"],
                                    weights=[1, 1, 1, 2]  # Favor 'erode'
                                )[0],
                                kernel_type=kernel_type,
                                kernel_shape=(
                                    (1, size) if random.choice([True, False]) else (size, 1)
                                ) if kernel_type == "ones" else (size, size)
                            )
                        )(kernel_type := random.choice(["ones", "upper_triangle", "lower_triangle", "x", "plus", "ellipse"])))
}

def _rand_phrase(text, w1=1, w2=3, min_len=3):
    words=[w.strip(string.punctuation+"“”’‘\"") for w in text.split()]
    words=[w for w in words if len(w)>=min_len]
    n=min(random.randint(w1,w2),len(words))
    return " ".join(random.sample(words,n)) if words else "अध्यायः"

def _rand_footnotes(text):
    return [_rand_phrase(text,4,8) for _ in range(random.randint(1,3))]

def _choose_font(dir_="../fonts"):
    return random.choice(list(pathlib.Path(dir_).glob("*.?tf"))).resolve()

def _jinja_env(dir_="templates"):
    return Environment(loader=FileSystemLoader(dir_),
                       autoescape=select_autoescape(["html","xml"]))

def _build_html(env, ctx):
    html = env.get_template("book_page.html.jinja").render(**ctx)
    reset = f'<style>@page{{counter-reset: page {ctx["page_start"]-1};}}</style>'
    return html.replace("</head>", reset + "</head>")

def _pdf_to_pngs(pdf_path, base_name, dpi=IMAGE_DPI):
    pngs = []
    for i, page in enumerate(convert_from_path(pdf_path, dpi=dpi)):
        fname = f"{base_name}_{i:02d}.png"
        page.save(fname, "PNG")
        pngs.append(pathlib.Path(fname))
    return pngs

def _choose_effects():
    chain = []
    for name, (prob, gen) in EFFECTS.items():
        if random.random() < prob:
            chain.append((getattr(dg, name), gen()))
    return chain

def _apply_chain(img, chain):
    for fn, params in chain:
        img = fn(img, **params)
    return img

def _degrade_and_cleanup(png_path):
    if random.random() < NO_DEGRADE_PROB:
        return png_path  # keep clean
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    chain = _choose_effects()
    if not chain:
        return png_path
    out = _apply_chain(img, chain)
    deg = png_path.with_name(png_path.stem + "-deg" + png_path.suffix)
    cv2.imwrite(str(deg), out)
    png_path.unlink()  # remove the clean version
    return deg

def gen_images(n):
    env = _jinja_env()
    final_images = []
    for i in tqdm(range(n)):
        mode = random.choices(LAYOUTS, weights=WEIGHTS, k=1)[0]
        w,h = (120,180)
        paras = [p.strip() for p in TEXT.split("\n\n") if p.strip()]
        ctx = dict(
            font_path=str(_choose_font()),
            chapter_title=_rand_phrase(TEXT,2,3),
            page_number_pos=random.choice(["top","bottom"]),
            page_side=random.choice(["left","right"]),
            mode=mode,
            paragraphs=paras,
            column_gap=random.randint(6,10),
            column_rule_width=round(random.uniform(0.3,0.6),2),
            subheading=_rand_phrase(TEXT,2,3),
            subhead_at=random.randint(1, max(1,len(paras)//2)),
            footnotes=_rand_footnotes(TEXT),
            margin_mm=random.randint(12,20),
            footnote_size=round(random.uniform(8,9),2),
            footnote_rule=round(random.uniform(0.8,1.2),2),
            font_size=round(random.uniform(8,12),2),
            line_height=round(random.uniform(1.25,1.5),2),
            header_size=round(random.uniform(10,12),2),
            weight_title=random.choice([500,700]),
            weight_body=random.choice([300,700]),
            weight_pageno=random.choice([400,500]),
            page_width=w, page_height=h,
            page_start=random.randint(1,999)
        )
        base = OUTPUT_DIR / f"{i:03d}-{mode}"
        pdf_file = str(base) + ".pdf"
        html = _build_html(env, ctx)
        HTML(string=html, base_url="templates").write_pdf(pdf_file)
        pngs = _pdf_to_pngs(pdf_file, str(base))
        pathlib.Path(pdf_file).unlink()
        for pg in pngs:
            final_images.append(_degrade_and_cleanup(pg))

    frames = []
    for p in sorted(final_images):
        img = imageio.imread(str(p))
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        frames.append(img)
    gif = OUTPUT_DIR / "showcase.gif"
    imageio.mimsave(str(gif), frames, duration=250)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="number of images to generate")
    args = parser.parse_args()
    gen_images(args.n)
