import argparse
import csv
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import face_recognition
import numpy as np
from sklearn.svm import SVC

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LFW_DIR = DATA_DIR / "lfw-deepfunneled" / "lfw-deepfunneled"
TRAIN_CSV = DATA_DIR / "peopleDevTrain.csv"
TEST_CSV = DATA_DIR / "peopleDevTest.csv"
MODEL_PATH = ROOT / "models" / "face_svc.pkl"
ENC_CACHE = ROOT / "models" / "train_encodings.pkl"
DEFAULT_TEST_DIR = ROOT / "test"
DEFAULT_TEST_OUT = ROOT / "models" / "test_results.csv"
DEFAULT_EVAL_FOLDER = ROOT / "test_100_peoples"

# =========================
# CẤU HÌNH FACE PIPELINE
# =========================
DETECT_MODEL = "cnn"         # "hog" hoặc "cnn"
LANDMARKS_MODEL = "large"    # "small" hoặc "large"
UPSAMPLE = 0                # 0 nhanh, tăng lên (1,2) bắt mặt nhỏ tốt hơn nhưng chậm


# =========================
# ĐỌC DANH SÁCH DỮ LIỆU
# =========================
def read_people(csv_path: Path) -> List[Tuple[str, int]]:
    """Đọc danh sách người và số lượng ảnh cần dùng từ file CSV."""
    people = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            count = int(row["images"])
            people.append((name, count))
    return people


def image_paths_for_person(name: str, count: int) -> List[str]:
    """Sinh danh sách đường dẫn ảnh cho một người theo định dạng LFW."""
    paths = []
    for idx in range(1, count + 1):
        filename = f"{name}_{idx:04d}.jpg"
        paths.append(str(LFW_DIR / name / filename))
    return paths


# =========================
# TIỀN XỬ LÝ ẢNH / ENCODING
# =========================
def _box_area(box: Tuple[int, int, int, int]) -> int:
    """Tính diện tích box (top, right, bottom, left)."""
    top, right, bottom, left = box
    return max(0, bottom - top) * max(0, right - left)


def encode_image(
    image_path: str,
    detect_model: str = DETECT_MODEL,
    landmarks_model: str = LANDMARKS_MODEL,
    upsample: int = UPSAMPLE,
    choose_largest_face: bool = True
) -> Optional[np.ndarray]:
    """
    Đọc ảnh và trả về encoding đầu tiên (hoặc mặt lớn nhất nếu choose_largest_face=True).

    Pipeline:
    - Kiểm tra file
    - Load ảnh
    - Detect face locations (hog/cnn)
    - (Tuỳ chọn) chọn 1 mặt lớn nhất nếu có nhiều mặt
    - Face encodings (embedding 128D) với landmarks model (small/large)
    """
    if not os.path.isfile(image_path):
        print(f"[BỎ QUA] Không tìm thấy ảnh: {image_path}")
        return None

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"[BỎ QUA] Lỗi đọc ảnh: {image_path} | {e}")
        return None

    # 1) Detect mặt (bounding boxes)
    try:
        locs = face_recognition.face_locations(
            image,
            number_of_times_to_upsample=upsample,
            model=detect_model
        )
    except Exception as e:
        print(f"[BỎ QUA] Lỗi detect face: {image_path} | {e}")
        return None

    if not locs:
        print(f"[BỎ QUA] Không tìm thấy khuôn mặt trong ảnh: {image_path}")
        return None

    # 2) Nếu có nhiều mặt: chọn mặt lớn nhất (thường là chủ thể)
    if choose_largest_face and len(locs) > 1:
        locs = [max(locs, key=_box_area)]

    # 3) Tạo embedding 128D dựa trên locations đã biết
    try:
        encs = face_recognition.face_encodings(
            image,
            known_face_locations=locs,
            model=landmarks_model  # "small" hoặc "large"
        )
    except Exception as e:
        print(f"[BỎ QUA] Lỗi tạo encodings: {image_path} | {e}")
        return None

    if not encs:
        print(f"[BỎ QUA] Không tạo được encoding cho ảnh: {image_path}")
        return None

    return encs[0]


def build_dataset(csv_path: Path):
    """Từ file CSV sinh X (encodings) và y (nhãn)."""
    people = read_people(csv_path)
    X, y = [], []
    for name, count in people:
        for img_path in image_paths_for_person(name, count):
            enc = encode_image(img_path)
            if enc is None:
                continue
            X.append(enc)
            y.append(name)

    if not X:
        raise RuntimeError(f"Không tạo được dữ liệu từ {csv_path}")

    return np.vstack(X), np.array(y)


# =========================
# HUẤN LUYỆN VÀ ĐÁNH GIÁ
# =========================
def train(train_csv: Path = TRAIN_CSV, model_path: Path = MODEL_PATH):
    """Huấn luyện SVC và lưu model."""
    print(f"[TRAIN] Đang xây dựng tập train từ {train_csv}")
    X_train, y_train = build_dataset(train_csv)
    print(f"[TRAIN] Tổng mẫu train: {len(y_train)}")

    clf = SVC(kernel="linear", probability=True, class_weight="balanced")
    clf.fit(X_train, y_train)
    print("[TRAIN] Huấn luyện xong.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[TRAIN] Đã lưu model vào {model_path}")

    # Lưu cache encodings để predict_dist khỏi phải build lại lâu
    ENC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(ENC_CACHE, "wb") as f:
        pickle.dump({"X_train": X_train, "y_train": y_train}, f)
    print(f"[TRAIN] Đã lưu cache encodings vào {ENC_CACHE}")


def evaluate(model_path: Path = MODEL_PATH, test_csv: Path = TEST_CSV):
    """Đánh giá model trên tập test."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Chưa có model {model_path}, hãy train trước.")

    print(f"[EVAL] Đang xây dựng tập test từ {test_csv}")
    X_test, y_test = build_dataset(test_csv)
    print(f"[EVAL] Tổng mẫu test: {len(y_test)}")

    with open(model_path, "rb") as f:
        clf: SVC = pickle.load(f)

    preds = clf.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"[EVAL] Accuracy (closed-set): {acc:.4f}")


def evaluate_folder(
    folder: Path = DEFAULT_EVAL_FOLDER,
    model_path: Path = MODEL_PATH,
    out_csv: Optional[Path] = None,
    upsample: int = UPSAMPLE,
):
    """Đánh giá model trên thư mục test có cấu trúc theo nhãn."""
    if not folder.exists():
        raise FileNotFoundError(f"Không thấy thư mục: {folder}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Chưa có model {model_path}, hãy train trước.")

    with open(model_path, "rb") as f:
        clf: SVC = pickle.load(f)

    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    rows = []
    total = 0
    correct = 0
    skipped = 0

    for person_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        label = person_dir.name
        for img_path in sorted(iter_image_files(person_dir)):
            total += 1
            enc = encode_image(str(img_path), upsample=upsample)
            if enc is None:
                skipped += 1
                rows.append({
                    "image_path": str(img_path),
                    "true_label": label,
                    "pred_label": "",
                    "pred_conf": "",
                    "status": "no_face",
                })
                continue
            pred_label = str(clf.predict([enc])[0])
            probs = clf.predict_proba([enc])[0]
            pred_conf = float(probs[class_to_idx[pred_label]])
            if pred_label == label:
                correct += 1
            rows.append({
                "image_path": str(img_path),
                "true_label": label,
                "pred_label": pred_label,
                "pred_conf": f"{pred_conf:.4f}",
                "status": "ok",
            })

    used = total - skipped
    acc = (correct / used) if used > 0 else 0.0
    print(f"[EVAL-FOLDER] Tổng ảnh: {total}, bỏ qua: {skipped}")
    print(f"[EVAL-FOLDER] Accuracy (closed-set): {acc:.4f}")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_path", "true_label", "pred_label", "pred_conf", "status"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[EVAL-FOLDER] Đã lưu kết quả vào {out_csv}")


def evaluate_folder_l2(
    folder: Path = DEFAULT_EVAL_FOLDER,
    out_csv: Optional[Path] = None,
    dist_threshold: float = 0.6,
    upsample: int = UPSAMPLE,
):
    """Đánh giá thư mục test theo nhãn thư mục con bằng khoảng cách L2."""
    if not folder.exists():
        raise FileNotFoundError(f"Không thấy thư mục: {folder}")

    cache = load_enc_cache()
    if cache:
        X_train, y_train = cache
    else:
        X_train, y_train = build_dataset(TRAIN_CSV)

    rows = []
    total = 0
    correct = 0
    skipped = 0

    for person_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        label = person_dir.name
        for img_path in sorted(iter_image_files(person_dir)):
            total += 1
            enc = encode_image(str(img_path), upsample=upsample)
            if enc is None:
                skipped += 1
                rows.append({
                    "image_path": str(img_path),
                    "true_label": label,
                    "pred_label": "",
                    "pred_dist": "",
                    "unknown": "",
                    "status": "no_face",
                })
                continue

            dists = np.linalg.norm(X_train - enc, axis=1)
            best_idx = int(np.argmin(dists))
            pred_label = y_train[best_idx]
            pred_dist = float(dists[best_idx])
            unknown = pred_dist > dist_threshold
            if pred_label == label:
                correct += 1
            rows.append({
                "image_path": str(img_path),
                "true_label": label,
                "pred_label": pred_label,
                "pred_dist": f"{pred_dist:.4f}",
                "unknown": str(unknown),
                "status": "ok",
            })

    used = total - skipped
    acc = (correct / used) if used > 0 else 0.0
    print(f"[EVAL-FOLDER-L2] Tổng ảnh: {total}, bỏ qua: {skipped}")
    print(f"[EVAL-FOLDER-L2] Accuracy (closed-set): {acc:.4f}")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["image_path", "true_label", "pred_label", "pred_dist", "unknown", "status"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[EVAL-FOLDER-L2] Đã lưu kết quả vào {out_csv}")


# =========================
# DỰ ĐOÁN ẢNH ĐƠN
# =========================
def predict_image(model_path: Path, image_path: str, unknown_threshold: float = 0.5):
    """Dự đoán nhãn cho một ảnh đơn lẻ (SVM + Unknown theo xác suất)."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Chưa có model {model_path}, hãy train trước.")
    with open(model_path, "rb") as f:
        clf: SVC = pickle.load(f)

    enc = encode_image(image_path)
    if enc is None:
        return

    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}
    pred_label = str(clf.predict([enc])[0])
    probs = clf.predict_proba([enc])[0]
    best_prob = float(probs[class_to_idx[pred_label]])

    if best_prob < unknown_threshold:
        print(f"Ảnh {image_path}: Unknown (conf={best_prob:.2f})")
    else:
        print(f"Ảnh {image_path}: {pred_label} (conf={best_prob:.2f})")


def load_enc_cache():
    """Load cache encodings nếu có."""
    if not ENC_CACHE.exists():
        return None
    try:
        with open(ENC_CACHE, "rb") as f:
            data = pickle.load(f)
        return data["X_train"], data["y_train"]
    except Exception:
        return None


def iter_image_files(folder: Path) -> Iterable[Path]:
    """Yield image files recursively under a folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def predict_folder(
    folder: Path,
    out_csv: Path,
    svm_threshold: float = 0.5,
    dist_threshold: float = 0.6,
    top_k: int = 1,
):
    """Predict all images in a folder and save results to CSV."""
    if not folder.exists():
        raise FileNotFoundError(f"Không thấy thư mục: {folder}")
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Chưa có model {MODEL_PATH}, hãy train trước.")

    with open(MODEL_PATH, "rb") as f:
        clf: SVC = pickle.load(f)
    class_to_idx = {c: i for i, c in enumerate(clf.classes_)}

    cache = load_enc_cache()
    if cache:
        X_train, y_train = cache
    else:
        X_train, y_train = build_dataset(TRAIN_CSV)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "status",
        "svm_label",
        "svm_conf",
        "svm_unknown",
        "dist_label",
        "dist_distance",
        "dist_unknown",
        "svm_threshold",
        "dist_threshold",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for img_path in sorted(iter_image_files(folder)):
            row = {
                "image_path": str(img_path),
                "status": "ok",
                "svm_label": "",
                "svm_conf": "",
                "svm_unknown": "",
                "dist_label": "",
                "dist_distance": "",
                "dist_unknown": "",
                "svm_threshold": svm_threshold,
                "dist_threshold": dist_threshold,
            }
            enc = encode_image(str(img_path))
            if enc is None:
                row["status"] = "no_face"
                writer.writerow(row)
                continue

            pred_label = str(clf.predict([enc])[0])
            probs = clf.predict_proba([enc])[0]
            best_prob = float(probs[class_to_idx[pred_label]])
            row["svm_label"] = pred_label
            row["svm_conf"] = f"{best_prob:.4f}"
            row["svm_unknown"] = str(best_prob < svm_threshold)

            dists = np.linalg.norm(X_train - enc, axis=1)
            best_idx = int(np.argmin(dists))
            best_dist = float(dists[best_idx])
            row["dist_label"] = y_train[best_idx]
            row["dist_distance"] = f"{best_dist:.4f}"
            row["dist_unknown"] = str(best_dist > dist_threshold)

            writer.writerow(row)
    print(f"[PRED-FOLDER] Đã lưu kết quả vào {out_csv}")


def predict_image_dist(image_path: str, distance_threshold: float = 0.6, top_k: int = 5):
    """
    Dự đoán bằng khoảng cách embedding (k-NN đơn giản).
    - Đọc cache encodings nếu có; nếu không sẽ build từ tập train.
    - Chọn nhãn có khoảng cách nhỏ nhất; nếu vượt ngưỡng -> Unknown.
    """
    enc = encode_image(image_path)
    if enc is None:
        return

    cache = load_enc_cache()
    if cache:
        print("[PRED-DIST] Đang dùng cache encodings.")
        X_train, y_train = cache
    else:
        print("[PRED-DIST] Đang tải tập train để so khớp khoảng cách (lâu)...")
        X_train, y_train = build_dataset(TRAIN_CSV)

    dists = np.linalg.norm(X_train - enc, axis=1)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_label = y_train[best_idx]

    if best_dist > distance_threshold:
        print(f"Ảnh {image_path}: Unknown (dist={best_dist:.3f}, thr={distance_threshold})")
    else:
        print(f"Ảnh {image_path}: {best_label} (dist={best_dist:.3f})")

    if top_k > 1:
        top_idx = np.argsort(dists)[:top_k]
        print("[PRED-DIST] Top gần nhất:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  {rank}. {y_train[idx]} | dist={float(dists[idx]):.3f}")


# =========================
# ĐÁNH GIÁ VERIFICATION (CẶP ẢNH)
# =========================
def read_pairs(pair_csv: Path, mismatch: bool = False):
    """Đọc file pairs. Trả về list tuple (path1, path2, is_match)."""
    pairs = []
    with open(pair_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Đọc dòng đầu, nếu là header thì bỏ qua, không thì xử lý như dữ liệu
        first = next(reader, None)
        if first is None:
            return pairs

        def is_data_row(row: List[str]) -> bool:
            try:
                if mismatch:
                    if len(row) != 4:
                        return False
                    int(row[1])
                    int(row[3])
                    return True
                if len(row) != 3:
                    return False
                int(row[1])
                int(row[2])
                return True
            except (TypeError, ValueError):
                return False

        rows_iter = reader
        if is_data_row(first):
            # first là dữ liệu
            rows_iter = [first] + list(reader)

        for row in rows_iter:
            if mismatch:
                if len(row) != 4:
                    continue
                name1, num1, name2, num2 = row
            else:
                if len(row) != 3:
                    continue
                name1, num1, num2 = row
                name2 = name1

            img1 = str(LFW_DIR / name1 / f"{name1}_{int(num1):04d}.jpg")
            img2 = str(LFW_DIR / name2 / f"{name2}_{int(num2):04d}.jpg")
            pairs.append((img1, img2, not mismatch))

    return pairs


def evaluate_pairs(threshold: float = 0.6, limit: Optional[int] = None):
    """Đánh giá theo nhiệm vụ verification (so khớp cặp ảnh) + cache encodings để chạy nhanh."""
    match_csv = DATA_DIR / "matchpairsDevTest.csv"
    mismatch_csv = DATA_DIR / "mismatchpairsDevTest.csv"
    match_pairs = read_pairs(match_csv, mismatch=False)
    mismatch_pairs = read_pairs(mismatch_csv, mismatch=True)

    if limit is not None and limit > 0:
        match_pairs = match_pairs[:limit]
        mismatch_pairs = mismatch_pairs[:limit]

    pairs = match_pairs + mismatch_pairs

    enc_cache: Dict[str, Optional[np.ndarray]] = {}

    def get_enc(path: str) -> Optional[np.ndarray]:
        if path not in enc_cache:
            enc_cache[path] = encode_image(path)
        return enc_cache[path]

    total = 0
    correct = 0
    skipped = 0

    for img1, img2, is_match in pairs:
        enc1 = get_enc(img1)
        enc2 = get_enc(img2)
        if enc1 is None or enc2 is None:
            skipped += 1
            continue

        dist = float(np.linalg.norm(enc1 - enc2))
        pred_match = dist < threshold
        if pred_match == is_match:
            correct += 1
        total += 1

    if total == 0:
        print("[EVAL-PAIRS] Không đủ dữ liệu để đánh giá.")
        return

    acc = correct / total
    print(f"[EVAL-PAIRS] Số cặp match: {len(match_pairs)}, mismatch: {len(mismatch_pairs)}")
    print(f"[EVAL-PAIRS] Tổng cặp dùng được: {total}, bỏ qua: {skipped}")
    print(f"[EVAL-PAIRS] Ngưỡng {threshold:.2f} | Accuracy: {acc:.4f}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Train/Test nhận diện khuôn mặt trên bộ LFW.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("train", help="Huấn luyện và lưu model SVC.")
    # Tạm không dùng:
    # sub.add_parser("eval", help="Đánh giá model (closed-set) trên DevTest.")
    eval_pairs_p = sub.add_parser("eval_pairs", help="Đánh giá verification bằng cặp match/mismatch.")
    eval_pairs_p.add_argument("--threshold", type=float, default=0.6, help="Ngưỡng khoảng cách cho match.")
    eval_pairs_p.add_argument("--limit", type=int, default=None, help="Chỉ dùng N dòng đầu mỗi file pairs.")

    pred_p = sub.add_parser("predict", help="Dự đoán một ảnh bất kỳ.")
    pred_p.add_argument("image", help="Đường dẫn ảnh cần nhận diện.")
    pred_p.add_argument("--threshold", type=float, default=0.5, help="Ngưỡng Unknown (0-1).")

    pred_dist = sub.add_parser("predict_dist", help="Dự đoán bằng khoảng cách embedding (k-NN đơn giản).")
    pred_dist.add_argument("image", help="Đường dẫn ảnh cần nhận diện.")
    pred_dist.add_argument("--threshold", type=float, default=0.6, help="Ngưỡng khoảng cách (L2).")
    pred_dist.add_argument("--topk", type=int, default=5, help="Hiển thị top-k lân cận để debug.")
    # Tạm không dùng:
    # pred_folder = sub.add_parser("predict_folder", help="Dự đoán tất cả ảnh trong một thư mục.")
    # pred_folder.add_argument("--dir", type=Path, default=DEFAULT_TEST_DIR, help="Thư mục ảnh cần nhận diện.")
    # pred_folder.add_argument("--out", type=Path, default=DEFAULT_TEST_OUT, help="File CSV kết quả.")
    # pred_folder.add_argument("--svm_threshold", type=float, default=0.5, help="Ngưỡng Unknown cho SVM.")
    # pred_folder.add_argument("--dist_threshold", type=float, default=0.6, help="Ngưỡng Unknown theo khoảng cách.")
    eval_folder_p = sub.add_parser("eval_folder", help="Đánh giá thư mục test theo nhãn thư mục con.")
    eval_folder_p.add_argument("--dir", type=Path, default=DEFAULT_EVAL_FOLDER, help="Thư mục test (mỗi nhãn 1 thư mục).")
    eval_folder_p.add_argument("--out", type=Path, default=None, help="File CSV kết quả (tuỳ chọn).")
    eval_folder_p.add_argument("--upsample", type=int, default=UPSAMPLE, help="Số lần upsample trước detect mặt.")
    eval_folder_l2_p = sub.add_parser("eval_folder_l2", help="Đánh giá thư mục test theo nhãn thư mục con bằng L2.")
    eval_folder_l2_p.add_argument("--dir", type=Path, default=DEFAULT_EVAL_FOLDER, help="Thư mục test (mỗi nhãn 1 thư mục).")
    eval_folder_l2_p.add_argument("--out", type=Path, default=None, help="File CSV kết quả (tuỳ chọn).")
    eval_folder_l2_p.add_argument("--dist_threshold", type=float, default=0.6, help="Ngưỡng Unknown theo khoảng cách.")
    eval_folder_l2_p.add_argument("--upsample", type=int, default=UPSAMPLE, help="Số lần upsample trước detect mặt.")

    args = parser.parse_args()

    if args.cmd == "train":
        train()
    # Tạm không dùng:
    # elif args.cmd == "eval":
    #     evaluate()
    elif args.cmd == "eval_pairs":
        evaluate_pairs(args.threshold, args.limit)
    elif args.cmd == "predict":
        predict_image(MODEL_PATH, args.image, args.threshold)
    elif args.cmd == "predict_dist":
        predict_image_dist(args.image, args.threshold, args.topk)
    # Tạm không dùng:
    # elif args.cmd == "predict_folder":
    #     predict_folder(args.dir, args.out, args.svm_threshold, args.dist_threshold)
    elif args.cmd == "eval_folder":
        evaluate_folder(args.dir, MODEL_PATH, args.out, args.upsample)
    elif args.cmd == "eval_folder_l2":
        evaluate_folder_l2(args.dir, args.out, args.dist_threshold, args.upsample)


if __name__ == "__main__":
    main()
