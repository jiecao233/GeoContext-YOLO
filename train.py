from ultralytics import YOLO

# model = YOLO("yolo26n-seg.yaml").load("yolo26n-seg.pt")
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     device=0,
#     workers=0,
#     cache=False,
#     project="runs/landslide",
#     name="yolo26n_seg_rgb"
# )

# model = YOLO("GeoContext_YOLO.yaml").load("yolo26n-seg.pt").cuda()
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     batch=64,
#     device=0,
#     workers=6,
#     project="runs/landslide",
#     name="GeoContext_YOLO"
# )

# model = YOLO("GeoContext_YOLO_LSK.yaml").load("yolo26n-seg.pt").cuda()
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     batch=64,
#     device=0,
#     workers=6,
#     project="runs/landslide",
#     name="GeoContext_YOLO_LSK"
# )

# model = YOLO("GeoContext_YOLO_DCN.yaml").load("yolo26n-seg.pt").cuda()
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     batch=64,
#     device=0,
#     workers=6,
#     project="runs/landslide",
#     name="GeoContext_YOLO_DCN"
# )

# model = YOLO("yolo12-seg.yaml").load("yolov12n-seg.pt").cuda()
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     batch=64,
#     device=0,
#     workers=6,
#     project="runs/landslide",
#     name="YOLO12-Seg"
# )

# model = YOLO("yolo11-seg.yaml").load("yolo11n-seg.pt").cuda()
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     batch=64,
#     device=0,
#     workers=6,
#     project="runs/landslide",
#     name="YOLO11-Seg"
# )

# model = YOLO("yolov8-seg.yaml").load("yolov8n-seg.pt").cuda()
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     batch=64,
#     device=0,
#     workers=6,
#     project="runs/landslide",
#     name="YOLOv8n-Seg"
# )

# model = YOLO("yolov10n-seg.yaml").load("yolov10n-seg.pt")
# model.train(
#     data="landslide_data/dataset_without_dem/landslide.yaml",
#     imgsz=640,
#     epochs=100,
#     device=0,
#     batch=64,
#     workers=6,
#     project="runs/landslide",
#     name="yolov10n-seg"
# )

model = YOLO("yolov9t-seg.yaml").load("yolov9t.pt")
model.train(
    data="landslide_data/dataset_without_dem/landslide.yaml",
    imgsz=640,
    epochs=100,
    device=0,
    batch=64,
    workers=6,
    project="runs/landslide",
    name="yolov9t-seg"
)