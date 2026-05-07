import os
import cv2
import numpy as np

import warnings

warnings.simplefilter("ignore")

data_root_folder = r"Data"
gt_root_folder = os.path.join(data_root_folder, "GT")
results_root_folder = os.path.join(data_root_folder, "Results")

TASK = "Degree"

TASK_FOLDERS = ["Class_classify", "Class", "Degree_classify", "Degree", "Style", "City"]
METHOD_NAMES = ["FFANet", "ChaIR", "CasDyF", "ConvIR", "SADNet", "SwinHistMap"]
DEGREE_CLASSES = ["Catastrophic", "Severe", "Slight"]
CLASS_CLASSES = [
    "Periodic Banding",
    "Pervasive Aging",
    "Homogeneous Bleaching",
    "Clouded Degradation",
    "Dense Granular Speckling",
]
STYLE_SUBS = ["1636", "1682", "grid", "youhua"]
CITY_NAMES = ["GuangZhou", "HongKong", "Tokyo"]


def print_message(message):
    print(message)


def calculate_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


def calculate_rmse(img1, img2):
    return np.sqrt(calculate_mse(img1, img2))


def calculate_nmse(img1, img2):
    mse = calculate_mse(img1, img2)
    var = np.var(img1.astype(np.float32))
    if var == 0:
        return float('inf')
    return mse / var


def calculate_mae(img1, img2):
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

def calculate_psnr_from_mse(mse_val, data_range=255.0):
    if mse_val <= 0:
        return 100.0
    return float(20.0 * np.log10(float(data_range) / float(np.sqrt(mse_val))))


def calculate_metrics(gt_path, pred_path):
    gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
    pred_img = cv2.imread(pred_path, cv2.IMREAD_COLOR)
    if gt_img is None or pred_img is None:
        return None, "Cannot read image"

    if gt_img.shape != pred_img.shape:
        return None, "Image dimensions do not match"

    gt_f = gt_img.astype(np.float32) / 255.0
    pred_f = pred_img.astype(np.float32) / 255.0

    mse_val = calculate_mse(gt_f, pred_f)
    mae_val = calculate_mae(gt_f, pred_f)
    rmse_val = np.sqrt(mse_val)
    psnr_val = float(cv2.PSNR(gt_img, pred_img))
    if np.isinf(psnr_val) or np.isnan(psnr_val):
        psnr_val = 100.0

    return {"mse": mse_val, "mae": mae_val, "rmse": rmse_val, "psnr": psnr_val}, None


def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))])


def evaluate_one_method(task_name, method_name):
    gt_folder = os.path.join(gt_root_folder, task_name)
    pred_folder = os.path.join(results_root_folder, task_name, method_name)

    if not os.path.isdir(gt_folder):
        return {"name": method_name, "error": f"GT folder does not exist: {gt_folder}"}
    if not os.path.isdir(pred_folder):
        return {"name": method_name, "error": f"Result folder does not exist: {pred_folder}"}

    gt_files = _list_images(gt_folder)
    pred_files = set(_list_images(pred_folder))
    common = [f for f in gt_files if f in pred_files]

    if not common:
        return {"name": method_name, "error": "No result images matched GT filenames"}

    mse_vals, mae_vals, rmse_vals, psnr_vals = [], [], [], []
    skipped = 0

    for fname in common:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)
        metrics, err = calculate_metrics(gt_path, pred_path)
        if err:
            skipped += 1
            continue
        mse_vals.append(metrics["mse"])
        mae_vals.append(metrics["mae"])
        rmse_vals.append(metrics["rmse"])
        psnr_vals.append(metrics["psnr"])

    if not mse_vals:
        return {"name": method_name, "error": f"All samples failed or were skipped (skipped={skipped})"}

    return {
        "name": method_name,
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "num_images": len(common),
        "num_skipped": skipped,
    }


def _get_classify_result_folder(task_name, class_name):
    folder = os.path.join(results_root_folder, task_name, class_name)
    return folder if os.path.isdir(folder) else None


def evaluate_classify_one_class(task_name, gt_task_name, class_name):
    gt_folder = os.path.join(gt_root_folder, gt_task_name)
    pred_folder = _get_classify_result_folder(task_name, class_name)

    if not os.path.isdir(gt_folder):
        return {"name": class_name, "error": f"GT folder does not exist: {gt_folder}"}
    if pred_folder is None:
        return {"name": class_name, "error": f"Result folder does not exist: {task_name}/{class_name}"}

    gt_files = _list_images(gt_folder)
    pred_files = set(_list_images(pred_folder))
    common = [f for f in gt_files if f in pred_files]

    if not common:
        return {"name": class_name, "error": "No result images matched GT filenames"}

    mse_vals, mae_vals, rmse_vals, psnr_vals = [], [], [], []
    skipped = 0

    for fname in common:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)
        metrics, err = calculate_metrics(gt_path, pred_path)
        if err:
            skipped += 1
            continue
        mse_vals.append(metrics["mse"])
        mae_vals.append(metrics["mae"])
        rmse_vals.append(metrics["rmse"])
        psnr_vals.append(metrics["psnr"])

    if not mse_vals:
        return {"name": class_name, "error": f"All samples failed or were skipped (skipped={skipped})"}

    return {
        "name": class_name,
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "num_images": len(common),
        "num_skipped": skipped,
    }

def evaluate_style_one_method(sub_name, method_name):
    gt_folder = os.path.join(gt_root_folder, "Style", sub_name)
    pred_folder = os.path.join(results_root_folder, "Style", method_name, sub_name)

    if not os.path.isdir(gt_folder):
        return {"name": method_name, "error": f"GT folder does not exist: {gt_folder}"}
    if not os.path.isdir(pred_folder):
        return {"name": method_name, "error": f"Result folder does not exist: {pred_folder}"}

    gt_files = _list_images(gt_folder)
    pred_files = set(_list_images(pred_folder))
    common = [f for f in gt_files if f in pred_files]

    if not common:
        return {"name": method_name, "error": "No result images matched GT filenames"}

    mse_vals, mae_vals, rmse_vals, psnr_vals = [], [], [], []
    skipped = 0

    for fname in common:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)
        metrics, err = calculate_metrics(gt_path, pred_path)
        if err:
            skipped += 1
            continue
        mse_vals.append(metrics["mse"])
        mae_vals.append(metrics["mae"])
        rmse_vals.append(metrics["rmse"])
        psnr_vals.append(metrics["psnr"])

    if not mse_vals:
        return {"name": method_name, "error": f"All samples failed or were skipped (skipped={skipped})"}

    return {
        "name": method_name,
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "num_images": len(common),
        "num_skipped": skipped,
    }

def evaluate_city_one_city(city_name):
    gt_folder = os.path.join(gt_root_folder, "City", city_name)
    pred_folder = os.path.join(results_root_folder, "City", city_name)

    if not os.path.isdir(gt_folder):
        return {"name": city_name, "error": f"GT folder does not exist: {gt_folder}"}
    if not os.path.isdir(pred_folder):
        return {"name": city_name, "error": f"Result folder does not exist: {pred_folder}"}

    gt_files = _list_images(gt_folder)
    pred_files = set(_list_images(pred_folder))
    common = [f for f in gt_files if f in pred_files]

    if not common:
        return {"name": city_name, "error": "No result images matched GT filenames"}

    mse_vals, mae_vals, rmse_vals, psnr_vals = [], [], [], []
    skipped = 0

    for fname in common:
        gt_path = os.path.join(gt_folder, fname)
        pred_path = os.path.join(pred_folder, fname)
        metrics, err = calculate_metrics(gt_path, pred_path)
        if err:
            skipped += 1
            continue
        mse_vals.append(metrics["mse"])
        mae_vals.append(metrics["mae"])
        rmse_vals.append(metrics["rmse"])
        psnr_vals.append(metrics["psnr"])

    if not mse_vals:
        return {"name": city_name, "error": f"All samples failed or were skipped (skipped={skipped})"}

    return {
        "name": city_name,
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "rmse": float(np.mean(rmse_vals)),
        "psnr": float(np.mean(psnr_vals)),
        "num_images": len(common),
        "num_skipped": skipped,
    }


def _print_table(task_name, rows):
    print_message(f"\n[{task_name}]")
    print_message(f"{'Name':>30}  {'MSE':<15}{'MAE':<15}{'RMSE':<15}{'PSNR':<15}")
    print_message("-" * 85)
    for r in rows:
        if "error" in r:
            print_message(f"{r['name']:>30}  {r['error']}")
        else:
            print_message(
                f"{r['name']:>30}  "
                f"{r['mse']:<15.8f}{r['mae']:<15.6f}{r['rmse']:<15.6f}{r['psnr']:<15.3f}"
            )


print("Calculating...")

if TASK not in TASK_FOLDERS:
    raise ValueError(f"TASK must be one of {TASK_FOLDERS}. Current: {TASK}")

if TASK == "Degree_classify":
    rows = [evaluate_classify_one_class("Degree_classify", "Degree", cls) for cls in DEGREE_CLASSES]
    _print_table(TASK, rows)
elif TASK == "Class_classify":
    rows = [evaluate_classify_one_class("Class_classify", "Class", cls) for cls in CLASS_CLASSES]
    _print_table(TASK, rows)
elif TASK == "Style":
    for sub in STYLE_SUBS:
        rows = [evaluate_style_one_method(sub, m) for m in METHOD_NAMES]
        _print_table(f"Style/{sub}", rows)
elif TASK == "City":
    rows = [evaluate_city_one_city(c) for c in CITY_NAMES]
    _print_table(TASK, rows)
else:
    rows = [evaluate_one_method(TASK, m) for m in METHOD_NAMES]
    _print_table(TASK, rows)

print("\nCalculation completed.")
