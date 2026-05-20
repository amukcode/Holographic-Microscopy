import numpy as np
import pandas as pd
from pathlib import Path

from skimage.transform import resize
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import convex_hull_image
from skimage.measure import label, regionprops, find_contours
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# ================= CONFIG =================
ROOT_DIR = Path("/Volumes/HD-PCFU3/Arushi, Oil/Data")
OUT_DIR = Path("/Users/arushimukherji/Desktop/AriyaLab/Processed_Data")
NPZ_DIR = OUT_DIR / "NPZ"
CSV_PATH = OUT_DIR / "May_all_particles.csv"

AMP_SUFFIX = "_Amp"
PHASE_SUFFIX = "_Pha"
TARGET_SHAPE = (64, 64)

SOLIDITY_MIN = 0.8

# ================= MATERIAL LABELS =================
OIL_MATERIALS = {"Canola", "Diesel", "Gasoline", "Bitumen", "Oil Sands", "Bitumen + Coke"}
BITUMENOUS_MATERIALS = {"Bitumen", "Oil Sands"}
MIXED_MATERIALS = {"Bitumen + Coke", "Oil Sands"}

# ================= HELPERS =================
def safe_load_txt(path):
    if path.name.startswith("._"):
        return None
    try:
        return np.loadtxt(path)
    except:
        return None

def minmax_normalize(arr):
    arr = arr.astype(np.float32)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return np.zeros_like(arr) if np.isclose(mx, mn) else (arr - mn) / (mx - mn)

# ================= SEGMENTATION =================
def build_mask_from_amp(amp):
    a = gaussian(amp.astype(np.float32), sigma=1.0, preserve_range=True)

    t = threshold_otsu(a)

    mask = a > t
    mask = binary_fill_holes(mask)

    lab = label(mask)
    regs = regionprops(lab)

    if not regs:
        return None, None, 0

    r = max(regs, key=lambda x: x.area)

    return lab == r.label, r, len(regs)

# ================= EQUIVALENT DIAMETER =================
def compute_equiv_diameter(area_px):
    return 2 * np.sqrt(area_px / np.pi)

# ================= CUT-OFF DETECTION =================
def compute_equiv_cutoff(vals):

    vals = np.array(vals)

    hist, edges = np.histogram(vals, bins=50)

    centers = (edges[:-1] + edges[1:]) / 2

    smooth = gaussian_filter1d(hist, sigma=2)

    peaks, _ = find_peaks(smooth)

    if len(peaks) > 0:

        main_peak = peaks[np.argmax(smooth[peaks])]

        if main_peak > 5:
            valley_idx = np.argmin(smooth[:main_peak])
            return centers[valley_idx]

    return np.percentile(vals, 10)

# ================= FERET =================
def compute_feret(mask):

    contours = find_contours(mask.astype(float), 0.5)

    if not contours:
        return np.nan, np.nan

    contour = max(contours, key=lambda x: len(x))

    coords = contour[:, ::-1]

    # max feret
    max_dist = 0

    for i in range(len(coords)):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        max_dist = max(max_dist, np.max(dists))

    # min feret via PCA
    centered = coords - np.mean(coords, axis=0)

    _, _, vh = np.linalg.svd(centered)

    axis = vh[1]

    proj = centered @ axis

    width = np.max(proj) - np.min(proj)

    return max_dist, width

# ================= INTENSITY =================
def compute_intensity_features(amp, mask):

    vals = amp[mask]

    if len(vals) < 10:
        return [np.nan] * 5

    mean = np.mean(vals)
    std = np.std(vals)

    p10, p90 = np.percentile(vals, [10, 90])

    contrast = p90 - p10

    dist = distance_transform_edt(mask)

    maxd = np.max(dist)

    if maxd < 1e-8:
        return [np.nan] * 5

    norm = dist / maxd

    edge = (norm < 0.2) & mask
    center = (norm > 0.6) & mask

    if np.sum(center) < 5:
        rim_ratio = np.nan
    else:
        rim_ratio = np.mean(amp[edge]) / (np.mean(amp[center]) + 1e-8)

    if np.sum(center) < 5 or np.sum(edge) < 5:
        radial_gradient = np.nan
    else:
        center_mean = np.mean(amp[center])
        edge_mean = np.mean(amp[edge])
        radial_gradient = center_mean - edge_mean

    return mean, std, contrast, rim_ratio, radial_gradient

# ================= PHASE =================
def compute_phase_features(pha, mask):

    vals = pha[mask]

    if len(vals) < 10:
        return [np.nan] * 5

    mean = np.mean(vals)
    std = np.std(vals)

    p10, p90 = np.percentile(vals, [10, 90])

    contrast = p90 - p10

    tau = (4.05e-7 / (2*np.pi)) * vals

    return mean, std, contrast, np.mean(tau), np.std(tau)

# ================= SHAPE =================
def compute_shape(mask, r):

    area = float(r.area)

    per = float(r.perimeter)

    hull = convex_hull_image(mask)

    hull_area = np.sum(hull)

    circ = 4 * np.pi * area / (per**2 + 1e-8)

    solidity = area / hull_area

    convex_dev = (hull_area - area) / hull_area

    fmax, fmin = compute_feret(mask)

    aspect = fmax / fmin if fmin > 0 else np.nan

    return area, circ, solidity, convex_dev, fmax, fmin, aspect

# ================= MAIN =================
def extract_particles():

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NPZ_DIR.mkdir(parents=True, exist_ok=True)

    # ===== PASS 1 =====
    print("\n=== COMPUTING EQUIV DIAMETER CUTS ===")

    per_material_cutoff = {}

    for mat in sorted(ROOT_DIR.iterdir()):

        if not mat.is_dir():
            continue

        material = mat.name

        d_vals = []

        for sub in mat.iterdir():

            if not sub.is_dir():
                continue

            for af in sorted(sub.glob(f"*{AMP_SUFFIX}*.txt")):
            
                if af.name.startswith("._"):
                    continue

                amp = safe_load_txt(af)

                if amp is None:
                    continue

                mask, r, n = build_mask_from_amp(amp)

                if mask is None or n > 1:
                    continue

                d_eq = compute_equiv_diameter(r.area)

                d_vals.append(d_eq)

        if len(d_vals) > 20:

            cutoff = compute_equiv_cutoff(d_vals)

            per_material_cutoff[material] = cutoff

            print(f"{material}: cutoff = {cutoff:.2f} px")

        else:
            print(f"{material}: insufficient data")

    # ===== PASS 2 =====
    rows = []

    stats_all = {}

    for mat in sorted(ROOT_DIR.iterdir()):

        if not mat.is_dir():
            continue

        material = mat.name

        cutoff = per_material_cutoff.get(material)

        stats = {
            "total": 0,
            "fragmented": 0,
            "below_cutoff": 0,
            "low_solidity": 0,
            "kept": 0
        }

        for sub in sorted(mat.iterdir()):

            if not sub.is_dir():
                continue

            for pid, af in enumerate(sorted(sub.glob(f"*{AMP_SUFFIX}*.txt"))):

                if af.name.startswith("._"):
                    continue

                stats["total"] += 1

                amp = safe_load_txt(af)

                pha = safe_load_txt(
                    af.with_name(
                        af.name.replace(AMP_SUFFIX, PHASE_SUFFIX)
                    )
                )

                if amp is None or pha is None:
                    continue

                mask, r, n = build_mask_from_amp(amp)

                if mask is None:
                    continue

                # fragmentation filter
                if n > 1:
                    stats["fragmented"] += 1
                    continue

                d_eq = compute_equiv_diameter(r.area)

                # cutoff filter
                if cutoff is not None and d_eq < cutoff:
                    stats["below_cutoff"] += 1
                    continue

                area, circ, solidity, convex_dev, fmax, fmin, aspect = compute_shape(mask, r)

                if solidity < SOLIDITY_MIN:
                    stats["low_solidity"] += 1
                    continue

                # ===== FEATURES =====
                amp_feats = compute_intensity_features(amp, mask)

                ph_feats = compute_phase_features(pha, mask)

                ptype = "single" if circ >= 0.6 else "aggregate"

                oil_id = int(material in OIL_MATERIALS)

                bitumen_id = int(material in BITUMENOUS_MATERIALS)

                mix_id = int(material in MIXED_MATERIALS)

                # ===== CROP TO PARTICLE =====
                minr, minc, maxr, maxc = r.bbox
                bbox_height = maxr - minr
                bbox_width = maxc - minc
                bbox_area = bbox_height * bbox_width

                amp_crop = amp[minr:maxr, minc:maxc]

                pha_crop = pha[minr:maxr, minc:maxc]

                npz_path = NPZ_DIR / f"{material}_{sub.name}_p{pid}.npz"
                particle_uid = npz_path.stem

                np.savez_compressed(
                    npz_path,

                    amplitude=resize(
                        minmax_normalize(amp_crop),
                        TARGET_SHAPE
                    ),

                    phase=resize(
                        minmax_normalize(pha_crop),
                        TARGET_SHAPE
                    ),
                )

                row = {
                    "particle_uid": particle_uid,
                    "material": material,
                    "acquisition_id": f"{material}_{sub.name}",
                    "frame_id": sub.name,
                    "particle_id": pid,
                    "npz_path": str(npz_path),

                    "particle_type": ptype,

                    "n_components": n,

                    "oil_id": oil_id,
                    "bitumen_id": bitumen_id,
                    "oil_mix_id": mix_id,

                    "bbox_height": bbox_height,
                    "bbox_width": bbox_width,
                    "bbox_area": bbox_area,

                    "area_px": area,
                    "equiv_diameter_px": d_eq,

                    "circularity": circ,
                    "solidity": solidity,
                    "convex_deviation": convex_dev,

                    "feret_max_px": fmax,
                    "feret_min_px": fmin,
                    "aspect_ratio": aspect,

                    "amp_mean": amp_feats[0],
                    "amp_std": amp_feats[1],
                    "amp_contrast": amp_feats[2],
                    "rim_ratio": amp_feats[3],
                    "radial_gradient": amp_feats[4],

                    "phase_mean": ph_feats[0],
                    "phase_std": ph_feats[1],
                    "phase_contrast": ph_feats[2],

                    "tau_mean": ph_feats[3],
                    "tau_std": ph_feats[4]
                }
                

                rows.append(row)

                stats["kept"] += 1

        stats_all[material] = stats

        print(f"\n{material} stats: {stats}")

    df = pd.DataFrame(rows)

    df.to_csv(CSV_PATH, index=False)

    print(f"\nSaved {len(df)} particles")

    return df

if __name__ == "__main__":
    extract_particles()

# import numpy as np
# import pandas as pd
# from pathlib import Path

# from skimage.transform import resize
# from skimage.filters import threshold_otsu, gaussian
# from skimage.morphology import convex_hull_image
# from skimage.measure import label, regionprops, find_contours
# from scipy.ndimage import binary_fill_holes, distance_transform_edt
# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import find_peaks

# # ================= CONFIG =================
# ROOT_DIR = Path("/Volumes/HD-PCFU3/Arushi, Oil/Data")
# OUT_DIR = Path("/Users/arushimukherji/Desktop/AriyaLab/Processed_Data")
# NPZ_DIR = OUT_DIR / "NPZ"
# CSV_PATH = OUT_DIR / "May_all_particles.csv"

# AMP_SUFFIX = "_Amp"
# PHASE_SUFFIX = "_Pha"
# TARGET_SHAPE = (64, 64)

# SOLIDITY_MIN = 0.8
# LARGEST_COMPONENT_RATIO_MIN = 0.7

# # ================= MATERIAL LABELS =================
# OIL_MATERIALS = {"Canola", "Diesel", "Gasoline", "Bitumen", "Oil Sands", "Bitumen + Coke"}
# BITUMENOUS_MATERIALS = {"Bitumen", "Oil Sands"}
# MIXED_MATERIALS = {"Bitumen + Coke", "Oil Sands"}

# # ================= HELPERS =================
# def safe_load_txt(path):
#     if path.name.startswith("._"):
#         return None
#     try:
#         return np.loadtxt(path)
#     except:
#         return None

# def minmax_normalize(arr):
#     arr = arr.astype(np.float32)
#     mn, mx = np.nanmin(arr), np.nanmax(arr)
#     return np.zeros_like(arr) if np.isclose(mx, mn) else (arr - mn) / (mx - mn)

# # ================= SEGMENTATION =================
# def build_mask_from_amp(amp):

#     a = gaussian(
#         amp.astype(np.float32),
#         sigma=1.0,
#         preserve_range=True
#     )

#     t = threshold_otsu(a)

#     mask = a > t

#     mask = binary_fill_holes(mask)

#     lab = label(mask)

#     regs = regionprops(lab)

#     if not regs:
#         return None, None, 0, np.nan

#     # --------------------------------------------------------
#     # component areas
#     # --------------------------------------------------------

#     areas = np.array([r.area for r in regs])

#     total_area = np.sum(areas)

#     largest_idx = np.argmax(areas)

#     largest_area = areas[largest_idx]

#     largest_ratio = largest_area / total_area

#     r = regs[largest_idx]

#     largest_mask = lab == r.label

#     return (
#         largest_mask,
#         r,
#         len(regs),
#         largest_ratio
#     )
# # ================= EQUIVALENT DIAMETER =================
# def compute_equiv_diameter(area_px):
#     return 2 * np.sqrt(area_px / np.pi)

# # ================= CUT-OFF DETECTION =================
# def compute_equiv_cutoff(vals):

#     vals = np.array(vals)

#     hist, edges = np.histogram(vals, bins=50)

#     centers = (edges[:-1] + edges[1:]) / 2

#     smooth = gaussian_filter1d(hist, sigma=2)

#     peaks, _ = find_peaks(smooth)

#     if len(peaks) > 0:

#         main_peak = peaks[np.argmax(smooth[peaks])]

#         if main_peak > 5:
#             valley_idx = np.argmin(smooth[:main_peak])
#             return centers[valley_idx]

#     return np.percentile(vals, 10)

# # ================= FERET =================
# def compute_feret(mask):

#     contours = find_contours(mask.astype(float), 0.5)

#     if not contours:
#         return np.nan, np.nan

#     contour = max(contours, key=lambda x: len(x))

#     coords = contour[:, ::-1]

#     # max feret
#     max_dist = 0

#     for i in range(len(coords)):
#         dists = np.linalg.norm(coords - coords[i], axis=1)
#         max_dist = max(max_dist, np.max(dists))

#     # min feret via PCA
#     centered = coords - np.mean(coords, axis=0)

#     _, _, vh = np.linalg.svd(centered)

#     axis = vh[1]

#     proj = centered @ axis

#     width = np.max(proj) - np.min(proj)

#     return max_dist, width

# # ================= INTENSITY =================
# def compute_intensity_features(amp, mask):

#     vals = amp[mask]

#     if len(vals) < 10:
#         return [np.nan] * 5

#     mean = np.mean(vals)
#     std = np.std(vals)

#     p10, p90 = np.percentile(vals, [10, 90])

#     contrast = p90 - p10

#     dist = distance_transform_edt(mask)

#     maxd = np.max(dist)

#     if maxd < 1e-8:
#         return [np.nan] * 5

#     norm = dist / maxd

#     edge = (norm < 0.2) & mask
#     center = (norm > 0.6) & mask

#     if np.sum(center) < 5:
#         rim_ratio = np.nan
#     else:
#         rim_ratio = np.mean(amp[edge]) / (np.mean(amp[center]) + 1e-8)

#     if np.sum(center) < 5 or np.sum(edge) < 5:
#         radial_gradient = np.nan
#     else:
#         center_mean = np.mean(amp[center])
#         edge_mean = np.mean(amp[edge])
#         radial_gradient = center_mean - edge_mean

#     return mean, std, contrast, rim_ratio, radial_gradient

# # ================= PHASE =================
# def compute_phase_features(pha, mask):

#     vals = pha[mask]

#     if len(vals) < 10:
#         return [np.nan] * 5

#     mean = np.mean(vals)
#     std = np.std(vals)

#     p10, p90 = np.percentile(vals, [10, 90])

#     contrast = p90 - p10

#     tau = (4.05e-7 / (2*np.pi)) * vals

#     return mean, std, contrast, np.mean(tau), np.std(tau)

# # ================= SHAPE =================
# def compute_shape(mask, r):

#     area = float(r.area)

#     per = float(r.perimeter)

#     hull = convex_hull_image(mask)

#     hull_area = np.sum(hull)

#     circ = 4 * np.pi * area / (per**2 + 1e-8)

#     solidity = area / hull_area

#     convex_dev = (hull_area - area) / hull_area

#     fmax, fmin = compute_feret(mask)

#     aspect = fmax / fmin if fmin > 0 else np.nan

#     return area, circ, solidity, convex_dev, fmax, fmin, aspect

# # ================= MAIN =================
# def extract_particles():

#     OUT_DIR.mkdir(parents=True, exist_ok=True)
#     NPZ_DIR.mkdir(parents=True, exist_ok=True)

#     # ===== PASS 1 =====
#     print("\n=== COMPUTING EQUIV DIAMETER CUTS ===")

#     per_material_cutoff = {}

#     for mat in ROOT_DIR.iterdir():

#         if not mat.is_dir():
#             continue

#         material = mat.name

#         d_vals = []

#         for sub in mat.iterdir():

#             if not sub.is_dir():
#                 continue

#             for af in sub.glob(f"*{AMP_SUFFIX}*.txt"):

#                 if af.name.startswith("._"):
#                     continue

#                 amp = safe_load_txt(af)

#                 if amp is None:
#                     continue

#                 mask, r, n, ratio = build_mask_from_amp(amp)

#                 if mask is None:
#                      continue
#                 if ratio < LARGEST_COMPONENT_RATIO_MIN:
#                      continue

#                 d_eq = compute_equiv_diameter(r.area)

#                 d_vals.append(d_eq)

#         if len(d_vals) > 20:

#             cutoff = compute_equiv_cutoff(d_vals)

#             per_material_cutoff[material] = cutoff

#             print(f"{material}: cutoff = {cutoff:.2f} px")

#         else:
#             print(f"{material}: insufficient data")

#     # ===== PASS 2 =====
#     rows = []

#     stats_all = {}

#     for mat in ROOT_DIR.iterdir():

#         if not mat.is_dir():
#             continue

#         material = mat.name

#         cutoff = per_material_cutoff.get(material)

#         stats = {
#             "total": 0,
#             "fragmented": 0,
#             "below_cutoff": 0,
#             "low_solidity": 0,
#             "kept": 0
#         }

#         for sub in mat.iterdir():

#             if not sub.is_dir():
#                 continue

#             for pid, af in enumerate(sub.glob(f"*{AMP_SUFFIX}*.txt")):

#                 if af.name.startswith("._"):
#                     continue

#                 stats["total"] += 1

#                 amp = safe_load_txt(af)

#                 pha = safe_load_txt(
#                     af.with_name(
#                         af.name.replace(AMP_SUFFIX, PHASE_SUFFIX)
#                     )
#                 )

#                 if amp is None or pha is None:
#                     continue

#                 mask, r, n, ratio = build_mask_from_amp(amp)

#                 if mask is None:
#                     continue

#                 # fragmentation filter
#                 if ratio < LARGEST_COMPONENT_RATIO_MIN:
#                     stats["fragmented"] += 1
#                     continue

#                 d_eq = compute_equiv_diameter(r.area)

#                 # cutoff filter
#                 if cutoff is not None and d_eq < cutoff:
#                     stats["below_cutoff"] += 1
#                     continue

#                 area, circ, solidity, convex_dev, fmax, fmin, aspect = compute_shape(mask, r)

#                 if solidity < SOLIDITY_MIN:
#                     stats["low_solidity"] += 1
#                     continue

#                 # ===== FEATURES =====
#                 amp_feats = compute_intensity_features(amp, mask)

#                 ph_feats = compute_phase_features(pha, mask)

#                 ptype = "single" if circ >= 0.6 else "aggregate"

#                 oil_id = int(material in OIL_MATERIALS)

#                 bitumen_id = int(material in BITUMENOUS_MATERIALS)

#                 mix_id = int(material in MIXED_MATERIALS)

#                 # ===== CROP TO PARTICLE =====
#                 minr, minc, maxr, maxc = r.bbox

#                 amp_crop = amp[minr:maxr, minc:maxc]

#                 pha_crop = pha[minr:maxr, minc:maxc]

#                 npz_path = NPZ_DIR / f"{material}_{sub.name}_p{pid}.npz"

#                 np.savez_compressed(
#                     npz_path,

#                     amplitude=resize(
#                         minmax_normalize(amp_crop),
#                         TARGET_SHAPE
#                     ),

#                     phase=resize(
#                         minmax_normalize(pha_crop),
#                         TARGET_SHAPE
#                     ),
#                 )

#                 row = {
#                     "material": material,
#                     "npz_path": str(npz_path),
#                     "frame_id": sub.name,
#                     "particle_id": pid,
#                     "particle_type": ptype,
#                     "n_components": n,
#                     "largest_component_ratio": ratio,
#                     "oil_id": oil_id,
#                     "bitumen_id": bitumen_id,
#                     "oil_mix_id": mix_id,
#                     "area_px": area,
#                     "equiv_diameter_px": d_eq,
#                     "circularity": circ,
#                     "solidity": solidity,
#                     "convex_deviation": convex_dev,
#                     "feret_max_px": fmax,
#                     "feret_min_px": fmin,
#                     "aspect_ratio": aspect,
#                     "amp_mean": amp_feats[0],
#                     "amp_std": amp_feats[1],
#                     "amp_contrast": amp_feats[2],
#                     "rim_ratio": amp_feats[3],
#                     "radial_gradient": amp_feats[4],
#                     "phase_mean": ph_feats[0],
#                     "phase_std": ph_feats[1],
#                     "phase_contrast": ph_feats[2],
#                     "tau_mean": ph_feats[3],
#                     "tau_std": ph_feats[4]
#                 }

#                 rows.append(row)

#                 stats["kept"] += 1

#         stats_all[material] = stats

#         print(f"\n{material} stats: {stats}")

#     df = pd.DataFrame(rows)

#     df.to_csv(CSV_PATH, index=False)

#     print(f"\nSaved {len(df)} particles")

#     return df

# if __name__ == "__main__":
#     extract_particles()
