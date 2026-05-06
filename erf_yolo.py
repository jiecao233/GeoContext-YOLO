import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def load_image(img_path: str, imgsz: int = 640) -> torch.Tensor:
    """Read image and convert to normalized tensor."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    tensor = torch.from_numpy(img).unsqueeze(0)  # 1, 3, H, W
    return tensor


def normalize_heatmap(hm: np.ndarray) -> np.ndarray:
    hm = hm - hm.min()
    hm = hm / (hm.max() + 1e-8)
    return hm


def get_module_by_name(model: torch.nn.Module, target_name: str):
    modules = dict(model.named_modules())
    if target_name not in modules:
        print("\nAvailable candidate modules:")
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if any(k in cls_name for k in ["C2PSA", "C2PSA_LSK", "C3k2", "C3k2_DCN", "C3k_DCN"]):
                print(f"{name}: {cls_name}")
        raise KeyError(f"Target module '{target_name}' not found.")
    return modules[target_name]


def compute_erf_for_image(
    model: torch.nn.Module,
    target_module: torch.nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    mode: str = "max",
) -> np.ndarray:
    """
    Compute ERF heatmap for one image.

    mode='center': use the center response of the target feature map.
    mode='max': use the spatial position with the strongest mean activation.
    """
    feature_container = {}

    def forward_hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        feature_container["feat"] = output

    hook = target_module.register_forward_hook(forward_hook)

    model.zero_grad(set_to_none=True)
    img_tensor = img_tensor.to(device)
    img_tensor.requires_grad_(True)

    _ = model(img_tensor)

    if "feat" not in feature_container:
        hook.remove()
        raise RuntimeError("Forward hook did not capture any feature.")

    feat = feature_container["feat"]  # [1, C, H, W]
    if feat.ndim != 4:
        hook.remove()
        raise RuntimeError(f"Expected 4D feature map, but got shape: {feat.shape}")

    _, c, h, w = feat.shape

    if mode == "center":
        yy, xx = h // 2, w // 2
    elif mode == "max":
        # Select the strongest spatial response.
        score_map = feat.detach().abs().mean(dim=1)[0]  # [H, W]
        idx = torch.argmax(score_map)
        yy = int(idx // w)
        xx = int(idx % w)
    else:
        hook.remove()
        raise ValueError("mode must be 'center' or 'max'.")

    # Use squared activation to avoid cancellation among channels.
    target_score = feat[0, :, yy, xx].pow(2).sum()

    model.zero_grad(set_to_none=True)
    target_score.backward()

    grad = img_tensor.grad.detach().abs().sum(dim=1)[0]  # [H, W]
    heatmap = grad.cpu().numpy()

    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3)
    heatmap = normalize_heatmap(heatmap)

    hook.remove()
    return heatmap


def save_heatmap(heatmap: np.ndarray, save_path: str):
    hm = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, hm_color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights.")
    parser.add_argument("--source", type=str, required=True, help="Image file or image folder.")
    parser.add_argument("--target", type=str, required=True, help="Target module name, e.g., model.10.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--mode", type=str, default="max", choices=["center", "max"])
    parser.add_argument("--out", type=str, default="erf_outputs")
    parser.add_argument("--print-modules", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(args.weights)
    model = yolo.model.to(device)
    model.eval()

    # Gradients are needed for ERF.
    for p in model.parameters():
        p.requires_grad_(True)

    if args.print_modules:
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__
            if any(k in cls_name for k in ["C2PSA", "C2PSA_LSK", "C3k2", "C3k2_DCN", "C3k_DCN"]):
                print(f"{name}: {cls_name}")
        return

    target_module = get_module_by_name(model, args.target)

    source = Path(args.source)
    if source.is_dir():
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]:
            image_paths.extend(source.glob(ext))
        image_paths = sorted(image_paths)[: args.max_images]
    else:
        image_paths = [source]

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    erf_sum = None
    valid_count = 0

    for img_path in image_paths:
        img_tensor = load_image(str(img_path), imgsz=args.imgsz)

        try:
            heatmap = compute_erf_for_image(
                model=model,
                target_module=target_module,
                img_tensor=img_tensor,
                device=device,
                mode=args.mode,
            )
        except Exception as e:
            print(f"[Skip] {img_path}: {e}")
            continue

        if erf_sum is None:
            erf_sum = np.zeros_like(heatmap, dtype=np.float64)

        erf_sum += heatmap
        valid_count += 1

        save_heatmap(
            heatmap,
            str(out_dir / f"{img_path.stem}_erf.png"),
        )

        print(f"[OK] {img_path.name}")

    if valid_count == 0:
        raise RuntimeError("No valid ERF results were generated.")

    avg_erf = erf_sum / valid_count
    avg_erf = normalize_heatmap(avg_erf)

    save_heatmap(avg_erf, str(out_dir / "average_erf.png"))
    np.save(out_dir / "average_erf.npy", avg_erf)

    print(f"\nSaved average ERF to: {out_dir / 'average_erf.png'}")
    print(f"Number of images used: {valid_count}")


if __name__ == "__main__":
    main()