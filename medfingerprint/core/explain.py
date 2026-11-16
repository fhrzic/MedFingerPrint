"""Plain-language explanations of features and outlier flags (no GUI dependencies)."""
from __future__ import annotations

# key -> (name, what it measures, likely causes when HIGH, likely causes when LOW)
FEATURES = {
    "mean": ("Mean intensity", "Average brightness (HU for CT) of the region.",
             "inverted image, overexposure, wrong windowing, contrast agent, mostly bone",
             "underexposure, blank/black image, data not in HU, mostly air or lung"),
    "p50": ("Median intensity", "Middle intensity value of the region: robust 'typical' brightness.",
            "inverted image, overexposure, wrong windowing",
            "underexposure, black image, data not in HU"),
    "std": ("Contrast (std)", "Spread of intensities. Low = flat image, high = strong contrast.",
            "noise, metal, sharp reconstruction kernel, mixed content",
            "blank or nearly uniform image, heavy smoothing"),
    "p00_5": ("Darkest 0.5 %", "Intensity below which the darkest 0.5 % of voxels lie.",
              "no dark background (cropped tight, inverted image)",
              "padding values (e.g. -2048/-3024 HU), very dark pixels, black borders"),
    "p99_5": ("Brightest 0.5 %", "Intensity above which the brightest 0.5 % of voxels lie.",
              "metal implants, contrast agent, burned-in text, saturation",
              "no bone/bright structures, underexposure, clipped range"),
    "skewness": ("Skewness", "Asymmetry of the intensity distribution (0 = symmetric).",
                 "few very bright structures on a darker image (bright tail)",
                 "few very dark regions on a bright image (dark tail), e.g. blank white image"),
    "entropy_bits": ("Grey-level variety (entropy)",
                     "How many different grey levels are used, in bits (8-bit max = 8).",
                     "very noisy image",
                     "blank, posterized, heavily compressed or almost empty image"),
    "frac_at_image_min": ("Share of pure-black pixels", "Fraction of pixels exactly at the image minimum.",
                          "black borders/padding, collimation, clipped dark values", ""),
    "frac_at_image_max": ("Share of pure-white pixels", "Fraction of pixels exactly at the image maximum.",
                          "saturation/overexposure, burned-in white areas, blank white image", ""),
    "frac_below_range": ("Share below histogram range",
                         "Fraction of voxels below -1024 HU (outside the normal CT range).",
                         "padding outside the scan field of view (-2048, -3024 HU), non-HU data", ""),
    "body_fraction": ("Body size in image", "Fraction of the image occupied by the body/foreground region.",
                      "tight crop, very large patient, bright background",
                      "small anatomy, large field of view, empty or nearly black image"),
    "spacing": ("Voxel/pixel spacing", "Physical size of a voxel in mm along one axis.",
                "coarser resolution, e.g. 5 mm thick slices instead of thin slices",
                "finer resolution, e.g. thin-slice or high-resolution reconstruction"),
    "shape": ("Image size", "Number of voxels/pixels along one axis.",
              "larger scan range or matrix", "short scan range, small or cropped image"),
    "extent_mm": ("Physical extent", "Scanned length in mm along one axis (shape x spacing).",
                  "long scan range (e.g. whole body)", "short scan range (e.g. single organ)"),
    "sharpness": ("Sharpness", "Edge strength (mean Laplacian, image resized to 512 px) relative to "
                  "the image contrast.",
                  "sharp reconstruction kernel, noise, fine texture, text/grids",
                  "blurry, smoothed, motion, upscaled or heavily compressed image"),
    "noise": ("Noise level", "Estimated pixel noise (Immerkaer) relative to the image contrast.",
              "noisy image: low dose, small detector signal, grain, sharp kernel",
              "very smooth image: denoised, smoothed, upscaled, synthetic"),
    "bright_background": ("Background brightness",
                          "Border brightness minus centre brightness, relative to the contrast. "
                          "X-rays normally have a dark background (negative values).",
                          "bright background: inverted grey scale, collimation, white padding",
                          "very dark background (normal for X-rays)"),
    "frac_border_padding": ("Uniform border / padding", "Share of the image in constant rows/columns "
                            "at the edges (black or white bars).",
                            "padding to a square, black bars, scanned film borders", ""),
    "colourfulness": ("Colour", "How much the RGB channels differ (0 = grey).",
                      "colour photo, coloured annotations or overlays", ""),
    "dynamic_range": ("Dynamic range used", "Range p0.5-p99.5 of the image divided by the range of its data "
                      "type (e.g. 0-255).", "", "flat, under-exposed or wrongly windowed image"),
    "frac_corner_bright": ("Bright marks in corners", "Share of corner pixels brighter than the 99.5th "
                           "percentile.", "burned-in text, L/R markers, rulers", ""),
    "histogram_w1": ("Whole histogram difference",
                     "How far the whole intensity histogram is from the dataset's typical histogram "
                     "(Wasserstein distance, in intensity units). Catches any change of overall look.",
                     "anything that changes the overall intensity distribution", ""),
}

AXES_3D = ("z (slices)", "y (rows)", "x (columns)")
AXES_2D = ("height", "width")


def parse_feature(feat: str):
    """'regions.body.mean' -> ('mean', 'body region'); 'geometry.spacing.0' -> ('spacing', 'axis 0')."""
    parts = feat.split(".")
    if parts[0] == "regions" and len(parts) >= 3:
        return parts[2], f"{parts[1]} region"
    if parts[0] == "geometry" and len(parts) >= 3 and parts[2].isdigit():
        return parts[1], int(parts[2])
    if parts[0] == "histogram_w1":
        return "histogram_w1", f"{parts[1]} region" if len(parts) > 1 else ""
    return parts[-1], ""


def feature_title(feat: str, dims: int = 3) -> str:
    key, where = parse_feature(feat)
    name = FEATURES.get(key, (key,))[0]
    if isinstance(where, int):
        axes = AXES_3D if dims == 3 else AXES_2D
        where = axes[where] if where < len(axes) else f"axis {where}"
    return f"{name} ({where})" if where else name


def feature_tooltip(feat: str, dims: int = 3) -> str:
    key, _ = parse_feature(feat)
    name, what, high, low = FEATURES.get(key, (key, "", "", ""))
    t = f"<b>{feature_title(feat, dims)}</b><br>{what}"
    if high:
        t += f"<br><b>Unusually high:</b> {high}"
    if low:
        t += f"<br><b>Unusually low:</b> {low}"
    return t + f"<br><span style='color:gray'>{feat}</span>"


def severity(score: float) -> str:
    a = abs(score)
    if a >= 15:
        return "far"
    if a >= 6:
        return "much"
    return "somewhat"


def obvious_problems(case: dict) -> list[str]:
    """Direct statements for images that are (almost) empty, black or white."""
    fl = case.get("flags", {}) or {}
    out = []
    bf = fl.get("body_fraction")
    if isinstance(bf, (int, float)) and bf < 0.01:
        out.append(f"! The body/foreground region covers only {100 * bf:.4g} % of the image: the image "
                   "is probably blank or empty. Body statistics then come from a few pixels and can "
                   "be misleading (e.g. look 'bright').")
    fmin, fmax = fl.get("frac_at_image_min"), fl.get("frac_at_image_max")
    if isinstance(fmin, (int, float)) and isinstance(fmax, (int, float)) and fmin >= 0.999 and fmax >= 0.999:
        out.append("! Every pixel has the same value: the image is completely blank (constant).")
        return out
    if isinstance(fmin, (int, float)) and fmin > 0.9:
        out.append(f"! {100 * fmin:.1f} % of all pixels have the minimum value: (almost) completely "
                   "black/empty image.")
    if isinstance(fmax, (int, float)) and fmax > 0.9:
        out.append(f"! {100 * fmax:.1f} % of all pixels have the maximum value: (almost) completely "
                   "white/saturated image.")
    if fl.get("not_hu_suspected"):
        out.append("! Values do not look like Hounsfield units (missing rescale or normalised data).")
    return out


def explain_flag(f: dict, dims: int = 3) -> str:
    """One plain-language sentence for one flagged feature."""
    key, _ = parse_feature(f["feature"])
    name, _, high, low = FEATURES.get(key, (key, "", "", ""))
    up = f["score"] > 0
    word = "higher" if up else "lower"
    sev = severity(f["score"])
    if key == "histogram_w1":
        word, sev = "different", {"far": "extremely", "much": "very"}.get(sev, sev)
        word += " from"
    else:
        word += " than"
    cause = high if up else low
    s = (f"• {feature_title(f['feature'], dims)} is {sev} {word} typical "
         f"({f['value']:.4g} vs median {f['dataset_median']:.4g}; score {f['score']:+.1f}).")
    if cause:
        s += f"\n    Possible causes: {cause}."
    return s


def verdict(n_features: int, max_score: float) -> str:
    if n_features >= 4 or max_score >= 50:
        return ("Unusual on many features or extremely far off: likely a broken or wrongly "
                "processed image (blank, noise, inverted, corrupted). Look at the preview.")
    if n_features >= 2:
        return ("Unusual on several features: check the preview; often a different protocol "
                "(kernel, slice thickness, scanner) or an image-quality problem.")
    return ("Unusual on one feature only: often normal variation. Check whether the value "
            "makes sense for this image.")
