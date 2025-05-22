import degradations as dg

import argparse, cv2, pathlib, random, string
import imageio.v2 as imageio
from jinja2 import Environment, FileSystemLoader, select_autoescape
import numpy as np
from pdf2image import convert_from_path
from tqdm import tqdm
from weasyprint import HTML

from .renderer_defaults import BOOK_PAGE_DEFAULTS, merge_params

DEFAULT_PARAMS = BOOK_PAGE_DEFAULTS

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

def _pdf_to_pngs(pdf_path, base_name, dpi=300):
    pngs = []
    for i, page in enumerate(convert_from_path(pdf_path, dpi=dpi)):
        fname = f"{base_name}_{i:02d}.png"
        page.save(fname, "PNG")
        pngs.append(pathlib.Path(fname))
    return pngs

def _choose_effects(effects):
    chain = []
    for name, (prob, gen) in effects.items():
        if random.random() < prob:
            chain.append((getattr(dg, name), gen()))
    return chain

def _apply_chain(img, chain):
    for fn, params in chain:
        img = fn(img, **params)
    return img

def _degrade_and_cleanup(png_path, no_degrade_prob, effects):
    if random.random() < no_degrade_prob:
        return png_path  # keep clean
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    chain = _choose_effects(effects)
    if not chain:
        return png_path
    out = _apply_chain(img, chain)
    deg = png_path.with_name(png_path.stem + "-deg" + png_path.suffix)
    cv2.imwrite(str(deg), out)
    png_path.unlink()  # remove the clean version
    return deg

def generate_book_pages(text, output_dir=None, params=None):
    """
    Generate book page images from Sanskrit text.
    
    Args:
        text (str): The Sanskrit text to render
        output_dir (str, optional): Directory to save images. If None, images are returned in memory
        params (dict, optional): Parameters to override defaults
        
    Returns:
        If output_dir is None: list of PIL.Image objects
        If output_dir is provided: None (images are saved to disk)
    """
    # Merge provided params with defaults
    params = merge_params(DEFAULT_PARAMS, params)
    
    # Create output directory if specified
    if output_dir:
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    env = _jinja_env()
    final_images = []
    
    # Split text into paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # Generate one page
    mode = random.choices(params['layouts'], weights=params['layout_weights'], k=1)[0]
    w,h = (120,180)
    
    ctx = dict(
        font_path=str(_choose_font()),
        chapter_title=_rand_phrase(text,2,3),
        page_number_pos=random.choice(["top","bottom"]),
        page_side=random.choice(["left","right"]),
        mode=mode,
        paragraphs=paras,
        column_gap=random.randint(6,10),
        column_rule_width=round(random.uniform(0.3,0.6),2),
        subheading=_rand_phrase(text,2,3),
        subhead_at=random.randint(1, max(1,len(paras)//2)),
        footnotes=_rand_footnotes(text),
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
    
    base = output_dir / "book_page" if output_dir else pathlib.Path("book_page")
    pdf_file = str(base) + ".pdf"
    html = _build_html(env, ctx)
    HTML(string=html, base_url="templates").write_pdf(pdf_file)
    
    pngs = _pdf_to_pngs(pdf_file, str(base), dpi=params['image_dpi'])
    pathlib.Path(pdf_file).unlink()
    
    for pg in pngs:
        final_image = _degrade_and_cleanup(pg, params['no_degrade_prob'], params['effects'])
        if output_dir:
            final_images.append(final_image)
        else:
            img = imageio.imread(str(final_image))
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            final_images.append(img)
            final_image.unlink()  # Clean up temporary file
    
    if output_dir:
        # Create showcase GIF
        frames = []
        for p in sorted(final_images):
            img = imageio.imread(str(p))
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[..., :3]
            frames.append(img)
        gif = output_dir / "showcase.gif"
        imageio.mimsave(str(gif), frames, duration=250)
        return None
    else:
        return final_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10,
                        help="number of images to generate")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_PARAMS['output_dir'],
                        help="output directory for generated images")
    args = parser.parse_args()
    
    # Read sample text
    text = open("sample_sa.txt", encoding="utf-8").read().strip()
    
    # Generate images
    generate_book_pages(text, output_dir=args.output_dir)
