import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = ROOT / "models" / "analysis"


def _box_area(box: Tuple[int, int, int, int]) -> int:
    top, right, bottom, left = box
    return max(0, bottom - top) * max(0, right - left)


def _load_image_rgb(image_path: str) -> np.ndarray:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Khong tim thay anh: {image_path}")
    return face_recognition.load_image_file(image_path)


def _detect_faces(
    image_rgb: np.ndarray,
    detect_model: str = "cnn",
    upsample: int = 0,
) -> List[Tuple[int, int, int, int]]:
    return face_recognition.face_locations(
        image_rgb,
        number_of_times_to_upsample=upsample,
        model=detect_model,
    )


def _encode_faces(
    image_rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    landmarks_model: str = "large",
) -> List[np.ndarray]:
    return face_recognition.face_encodings(
        image_rgb,
        known_face_locations=boxes,
        model=landmarks_model,
    )


def _collect_landmarks(
    image_rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    landmarks_model: str = "large",
) -> List[Dict[str, List[Tuple[int, int]]]]:
    return face_recognition.face_landmarks(
        image_rgb,
        face_locations=boxes,
        model=landmarks_model,
    )


def _draw_analysis(
    image_rgb: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    landmarks: List[Dict[str, List[Tuple[int, int]]]],
    out_path: Path,
) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(boxes):
        top, right, bottom, left = box
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(
            image_bgr,
            f"face_{i}",
            (left, max(20, top - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if i < len(landmarks):
            for _, pts in landmarks[i].items():
                for (x, y) in pts:
                    cv2.circle(image_bgr, (x, y), 1, (0, 0, 255), -1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image_bgr)


def inspect_image(
    image_path: str,
    detect_model: str = "cnn",
    landmarks_model: str = "large",
    upsample: int = 0,
    choose_largest_face: bool = False,
    out_image: Optional[Path] = None,
    out_json: Optional[Path] = None,
) -> None:
    image_rgb = _load_image_rgb(image_path)
    boxes = _detect_faces(image_rgb, detect_model=detect_model, upsample=upsample)

    if not boxes:
        print("[ANALYZE] Khong detect duoc khuon mat.")
        return

    if choose_largest_face and len(boxes) > 1:
        boxes = [max(boxes, key=_box_area)]

    landmarks = _collect_landmarks(image_rgb, boxes, landmarks_model=landmarks_model)
    encodings = _encode_faces(image_rgb, boxes, landmarks_model=landmarks_model)

    report = {
        "image_path": image_path,
        "detect_model": detect_model,
        "landmarks_model": landmarks_model,
        "upsample": upsample,
        "num_faces": len(boxes),
        "faces": [],
    }

    print(f"[ANALYZE] So khuon mat: {len(boxes)}")
    for i, box in enumerate(boxes):
        top, right, bottom, left = box
        w = right - left
        h = bottom - top

        face_info = {
            "face_index": i,
            "box": {"top": top, "right": right, "bottom": bottom, "left": left},
            "width": w,
            "height": h,
            "landmark_parts": {},
            "encoding": {},
        }

        if i < len(landmarks):
            total_pts = 0
            for part, pts in landmarks[i].items():
                face_info["landmark_parts"][part] = len(pts)
                total_pts += len(pts)
            face_info["landmark_total_points"] = total_pts
        else:
            face_info["landmark_total_points"] = 0

        if i < len(encodings):
            enc = encodings[i]
            face_info["encoding"] = {
                "shape": list(enc.shape),
                "l2_norm": float(np.linalg.norm(enc)),
                "min": float(np.min(enc)),
                "max": float(np.max(enc)),
                "mean": float(np.mean(enc)),
                "std": float(np.std(enc)),
                "first_10": [float(x) for x in enc[:10]],
            }

            print(
                f"[FACE {i}] box=({left},{top},{right},{bottom}) "
                f"landmarks={face_info['landmark_total_points']} "
                f"embedding_shape={enc.shape} norm={face_info['encoding']['l2_norm']:.4f}"
            )
        else:
            print(f"[FACE {i}] Khong tao duoc embedding.")

        report["faces"].append(face_info)

    if out_image is None:
        stem = Path(image_path).stem
        out_image = DEFAULT_OUT_DIR / f"{stem}_analysis.jpg"

    _draw_analysis(image_rgb, boxes, landmarks, out_image)
    print(f"[ANALYZE] Da luu anh bbox+landmarks: {out_image}")

    if out_json is None:
        stem = Path(image_path).stem
        out_json = DEFAULT_OUT_DIR / f"{stem}_analysis.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(f"[ANALYZE] Da luu report JSON: {out_json}")



def compare_two(
    image1: str,
    image2: str,
    threshold: float = 0.6,
    detect_model: str = "cnn",
    landmarks_model: str = "large",
    upsample: int = 0,
) -> None:
    def first_encoding(path: str) -> Optional[np.ndarray]:
        image_rgb = _load_image_rgb(path)
        boxes = _detect_faces(image_rgb, detect_model=detect_model, upsample=upsample)
        if not boxes:
            return None
        encs = _encode_faces(image_rgb, [boxes[0]], landmarks_model=landmarks_model)
        if not encs:
            return None
        return encs[0]

    enc1 = first_encoding(image1)
    enc2 = first_encoding(image2)

    if enc1 is None or enc2 is None:
        print("[COMPARE] Khong tao duoc encoding cho mot trong hai anh.")
        return

    dist = float(np.linalg.norm(enc1 - enc2))
    is_match = dist < threshold

    print(f"[COMPARE] distance_L2 = {dist:.6f}")
    print(f"[COMPARE] threshold   = {threshold:.3f}")
    print(f"[COMPARE] predict     = {'MATCH' if is_match else 'MISMATCH'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phan tich sau pipeline detect/landmarks/embedding 128D cua face_recognition."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect_image", help="Phan tich 1 anh: bbox + landmarks + embedding stats.")
    inspect_p.add_argument("--image", required=True, help="Duong dan anh can phan tich.")
    inspect_p.add_argument("--detect-model", default="cnn", choices=["cnn", "hog"], help="Model detect mat.")
    inspect_p.add_argument("--landmarks-model", default="large", choices=["large", "small"], help="Model landmarks/encoding.")
    inspect_p.add_argument("--upsample", type=int, default=0, help="So lan upsample truoc detect.")
    inspect_p.add_argument("--choose-largest-face", action="store_true", help="Chi giu mat lon nhat neu co nhieu mat.")
    inspect_p.add_argument("--out-image", type=Path, default=None, help="File anh output da ve bbox+landmarks.")
    inspect_p.add_argument("--out-json", type=Path, default=None, help="File JSON thong ke embedding/landmarks.")

    cmp_p = sub.add_parser("compare_two", help="So sanh 2 anh bang L2 distance.")
    cmp_p.add_argument("--image1", required=True, help="Anh thu nhat.")
    cmp_p.add_argument("--image2", required=True, help="Anh thu hai.")
    cmp_p.add_argument("--threshold", type=float, default=0.6, help="Nguong match theo L2.")
    cmp_p.add_argument("--detect-model", default="cnn", choices=["cnn", "hog"], help="Model detect mat.")
    cmp_p.add_argument("--landmarks-model", default="large", choices=["large", "small"], help="Model landmarks/encoding.")
    cmp_p.add_argument("--upsample", type=int, default=0, help="So lan upsample truoc detect.")

    args = parser.parse_args()

    if args.cmd == "inspect_image":
        inspect_image(
            image_path=args.image,
            detect_model=args.detect_model,
            landmarks_model=args.landmarks_model,
            upsample=args.upsample,
            choose_largest_face=args.choose_largest_face,
            out_image=args.out_image,
            out_json=args.out_json,
        )
    elif args.cmd == "compare_two":
        compare_two(
            image1=args.image1,
            image2=args.image2,
            threshold=args.threshold,
            detect_model=args.detect_model,
            landmarks_model=args.landmarks_model,
            upsample=args.upsample,
        )


if __name__ == "__main__":
    main()
