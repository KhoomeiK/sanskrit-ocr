#!/usr/bin/env python3
import os
import glob
from fontTools.ttLib import TTFont
from collections import defaultdict

def check_devanagari_support(font_path):
    try:
        font = TTFont(font_path)
        
        cmap = font.getBestCmap()
        
        devanagari_range = range(0x0900, 0x097F + 1)
        supported_chars = [hex(c) for c in devanagari_range if c in cmap]
        
        support_percentage = len(supported_chars) / len(devanagari_range) * 100
        
        return {
            "supports_devanagari": len(supported_chars) > 0,
            "coverage_percentage": support_percentage,
            "supported_chars": len(supported_chars),
            "total_devanagari_chars": len(devanagari_range)
        }
    except Exception as e:
        return {
            "error": str(e),
            "supports_devanagari": False,
            "coverage_percentage": 0
        }

def find_all_fonts(base_dir="datagen/fonts"):
    font_extensions = ['.ttf', '.otf', '.woff', '.woff2', '.TTF', '.OTF', '.pfb', '.PFB']
    font_files = []
    
    for ext in font_extensions:
        font_files.extend(glob.glob(f"{base_dir}/**/*{ext}", recursive=True))
    
    return font_files

def shorten_family_name(name, max_length=25):
    if len(name) <= max_length:
        return name
    
    if "uttara-chandas" in name:
        return "uttara-chandas"
    
    return name[:max_length-3] + "..."

def print_summary(results):
    print("\n" + "=" * 80)
    print("SANSKRIT FONT ANALYSIS SUMMARY")
    print("=" * 80)
    
    families = defaultdict(list)
    for r in results:
        path = r["path"]
        parts = path.split('/')
        
        if len(parts) > 2:
            # Use the actual font name instead of the directory name
            font_filename = os.path.basename(path)
            family = os.path.splitext(font_filename)[0]
        else:
            filename = os.path.basename(path)
            family = os.path.splitext(filename)[0]
        
        families[family].append(r)
    
    family_stats = {}
    for family, fonts in families.items():
        supporting_fonts = [f for f in fonts if f["supports_devanagari"]]
        if supporting_fonts:
            avg_coverage = sum(f["coverage_percentage"] for f in supporting_fonts) / len(supporting_fonts)
            family_stats[family] = {
                "fonts": len(fonts),
                "supporting": len(supporting_fonts),
                "avg_coverage": avg_coverage
            }
    
    sorted_families = sorted(
        family_stats.items(),
        key=lambda x: x[1]["avg_coverage"],
        reverse=True
    )
    
    col_family = 26
    col_fonts = 6
    col_support = 10
    col_coverage = 13
    col_chars = 10
    
    print(f"{'Font Family':<{col_family}} {'Fonts':^{col_fonts}} {'Supporting':^{col_support}} {'Avg Coverage':^{col_coverage}} {'Characters':^{col_chars}}")
    print("-" * 80)
    
    for family, stats in sorted_families:
        fonts = families[family]
        supporting_fonts = [f for f in fonts if f["supports_devanagari"]]
        if supporting_fonts:
            char_count = f"{supporting_fonts[0]['supported_chars']}/{supporting_fonts[0]['total_devanagari_chars']}"
        else:
            char_count = "0/128"
        
        short_family = shorten_family_name(family)
        print(f"{short_family:<{col_family}} {stats['fonts']:^{col_fonts}} {stats['supporting']:^{col_support}} {stats['avg_coverage']:^{col_coverage}.1f} {char_count:^{col_chars}}")
    
    total_fonts = len(results)
    supporting_fonts = sum(1 for r in results if r.get("supports_devanagari", False))
    error_fonts = sum(1 for r in results if "error" in r and r["error"] is not None)
    
    avg_coverage = 0
    if supporting_fonts > 0:
        avg_coverage = sum(r["coverage_percentage"] for r in results if r.get("supports_devanagari", False)) / supporting_fonts
    
    print("\n" + "=" * 80)
    print(f"Total fonts analyzed: {total_fonts}")
    print(f"Fonts supporting Sanskrit (Devanagari): {supporting_fonts} ({supporting_fonts/total_fonts*100:.1f}%)")
    print(f"Fonts with analysis errors: {error_fonts}")
    print(f"Average Devanagari coverage: {avg_coverage:.1f}%")
    print("=" * 80)
    
    # Print fonts suitable for use with render_text.py
    if supporting_fonts > 0:
        print("\nFonts you can use with render_text.py:")
        for family, stats in sorted_families[:5]:  # Show top 5 families
            fonts = families[family]
            supporting_fonts = [f for f in fonts if f["supports_devanagari"]]
            for font in supporting_fonts[:2]:  # Show first 2 fonts per family
                font_path = font["path"]
                font_name = os.path.basename(font_path)
                coverage = font["coverage_percentage"]
                print(f"  {font_name} - {coverage:.1f}% coverage")
                print(f"    Command: python3 datagen/augmentations/render_text.py --font \"{font_name}\"")

def main():
    font_files = find_all_fonts()
    results = []
    
    print(f"Found {len(font_files)} font files")
    print("-" * 80)
    
    for font_path in font_files:
        result = check_devanagari_support(font_path)
        
        if "error" in result:
            print(f"Error analyzing {font_path}: {result['error']}")
            result["path"] = font_path
            results.append(result)
            continue
        
        supports = "✓" if result["supports_devanagari"] else "✗"
        coverage = f"{result['coverage_percentage']:.1f}%"
        chars = f"{result['supported_chars']}/{result['total_devanagari_chars']}"
        
        print(f"{supports} {font_path} - {coverage} ({chars} Devanagari chars)")
        
        results.append({
            "path": font_path,
            "supports_devanagari": result["supports_devanagari"],
            "coverage_percentage": result["coverage_percentage"],
            "supported_chars": result["supported_chars"],
            "total_devanagari_chars": result["total_devanagari_chars"]
        })
    
    print("-" * 80)
    supporting_fonts = [r for r in results if r["supports_devanagari"]]
    print(f"Summary: {len(supporting_fonts)}/{len(results)} fonts support Devanagari (Sanskrit)")
    
    print_summary(results)

if __name__ == "__main__":
    main() 