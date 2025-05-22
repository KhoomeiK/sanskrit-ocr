import argparse, random, pathlib, cv2, numpy as np, imageio.v2 as imageio, string
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML
from pdf2image import convert_from_path
import degradations as dg
from tqdm import tqdm

from .renderer_defaults import PARCHMENT_DEFAULTS, merge_params

DEFAULT_PARAMS = PARCHMENT_DEFAULTS

DEV = "०१२३४५६७८९"
def _sa_num(n):
    return "".join(DEV[int(d)] for d in str(n))

def _choose_font(dir_="../fonts"):
    files = list(pathlib.Path(dir_).glob("*.otf")) + list(pathlib.Path(dir_).glob("*.ttf"))
    return str(random.choice(files).resolve())

def _env():
    return Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(["html", "xml"])
    )

def _random_chain(effects):
    chain = []
    for name, (prob, gen) in effects.items():
        if random.random() < prob:
            chain.append((getattr(dg, name), gen()))
    return chain

def _apply_chain(img, chain):
    for fn, kw in chain:
        img = fn(img, **kw)
    return img

def _degrade(p, no_degrade_prob, effects):
    """
    Degrade the image at Path p and return (output_path, chain_applied).
    If no degradation or NO_DEG hit, returns (p, []).
    """
    if random.random() < no_degrade_prob:
        return p, []

    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    chain = _random_chain(effects)
    if not chain:
        return p, []

    out = _apply_chain(img, chain)
    deg = p.with_name(p.stem + "-deg.png")
    cv2.imwrite(str(deg), out)
    p.unlink()
    return deg, chain

def _pdf2png(pdf, stem, dpi):
    pages = []
    for i, pg in enumerate(convert_from_path(pdf, dpi=dpi)):
        fn = f"{stem}_{i:02d}.png"
        pg.save(fn, "PNG")
        pages.append(pathlib.Path(fn))
    return pages

def generate_parchment_leaves(text, output_dir=None, params=None):
    """
    Generate parchment leaf images from Sanskrit text.
    
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
    
    env = _env()
    frames = []
    raw_paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Open degradation log if output_dir is provided
    log_path = output_dir / "degradation_log.txt" if output_dir else None
    log_f = log_path.open("w", encoding="utf-8") if log_path else None

    try:
        margin = round(random.uniform(8, 15), 2)
        stripe = round(random.uniform(0.5, 0.8), 2)
        gap = round(random.uniform(0.2, 0.6), 2)
        innerpad = round(stripe + random.uniform(2, 4), 2)
        max_lh = random.uniform(1.25, 1.4)
        line_h = random.uniform(1.0, max_lh)
        para_spc = round(random.uniform(0.5, 2.0), 2)
        para_scale = round(random.uniform(1.1, 1.4), 2)

        inner_left = margin + stripe + gap
        safe_text_mm = margin + stripe + gap + stripe
        page_no_mm = margin / 2

        paragraphs = []
        for ptext in raw_paras:
            paragraphs.append({
                "text": ptext,
                "large": random.random() < 0.25
            })

        ctx = {
            "font_path": _choose_font(),
            "parchment": random.choice(params['parchment_colors']),
            "ink": random.choice(params['ink_colors']),

            "page_width": 180,
            "page_height": 70,
            "font_size": random.uniform(6, 9),
            "line_height": line_h,

            "paragraphs": paragraphs,
            "margin_mm": margin,
            "stripe_thick": stripe,
            "inner_left_mm": inner_left,
            "inner_right_mm": inner_left,
            "text_margin_mm": safe_text_mm,
            "stripe_gap": gap,
            "inner_pad": innerpad,

            "para_spacing": para_spc,
            "paragraph_scale": para_scale,

            "page_no_sa": _sa_num(random.randint(1, 999)),
            "show_page_no": random.random() < 0.6,
            "page_side": random.choice(["left", "right"]),
            "page_no_mm": page_no_mm,
        }

        html = env.get_template("parchment_leaf.html.jinja").render(**ctx)
        base = output_dir / "leaf" if output_dir else pathlib.Path("leaf")
        pdf = str(base) + ".pdf"
        HTML(string=html, base_url="templates").write_pdf(pdf)

        for page_idx, p in enumerate(_pdf2png(pdf, str(base), params['dpi'])):
            out_path, chain = _degrade(p, params['no_degrade_prob'], params['effects'])
            
            if output_dir:
                frames.append(imageio.imread(str(out_path)))
            else:
                img = imageio.imread(str(out_path))
                if img.ndim == 2:
                    img = np.stack([img] * 3, -1)
                if img.shape[2] == 4:
                    img = img[..., :3]
                frames.append(img)
                out_path.unlink()  # Clean up temporary file

            # Build human-readable description
            if log_f:
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

        if output_dir:
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
            imageio.mimsave(str(output_dir / "showcase.gif"), norm, duration=250)
            return None
        else:
            return frames

    finally:
        if log_f:
            log_f.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=10)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_PARAMS['output_dir'])
    args = ap.parse_args()
    
    # Read sample text
    text = pathlib.Path("sample_sa.txt").read_text("utf-8").strip()
    
    # Generate images
    generate_parchment_leaves(text, output_dir=args.output_dir)