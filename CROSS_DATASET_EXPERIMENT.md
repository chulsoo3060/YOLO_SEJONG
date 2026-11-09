# YOLO11n Cross-Dataset Evaluation (3 bottle datasets)

This experiment fine-tunes **three independent YOLO11n models** (one for each bottle dataset), and evaluates each model on all three held-out test sets (nine evaluations). All runs start from the same COCO-pretrained `yolo11n.pt` weights and use 50 epochs, `imgsz=640`, `batch=16`, `seed=42` in the Colab notebook.

## Files

- `YOLO_3Models_Cross_Dataset_Colab.ipynb` — Colab notebook for training and evaluation (requires uploading the three dataset ZIPs).
- `results/cross_dataset/cross_dataset_results.csv` — 9 evaluation rows, including precision, recall, mAP50, mAP50-95, and inference time.
- `results/cross_dataset/self_test_performance.png` — models evaluated on their own test sets.
- `results/cross_dataset/cross_dataset_heatmap.png` — 3 × 3 mAP50-95 comparison.

## mAP50-95 (%) — Model × Test Dataset

| Trained model | Mine test | Sanghyeon test | Jian test |
|---|---:|---:|---:|
| Mine | 89.88 | 10.79 | 0.03 |
| Sanghyeon | 5.84 | 83.59 | 21.11 |
| Jian | 0.07 | 5.22 | 80.31 |

Each model performs much better on its own test set than on the other two. This suggests sensitivity to dataset differences, but the effects of object appearance, camera, background, lighting, distance, and annotation practices have not been isolated. Test sets contain only 15–20 images, so reported results have substantial uncertainty.

## Reproducibility and privacy

The three datasets and trained `.pt` weights are **not included** here. Upload the dataset ZIPs in Colab to rerun training. Avoid committing raw webcam imagery containing people or personally identifiable information without permission. To reproduce the reported metrics exactly, use the saved models from the original run; retraining may yield slightly different measurements.

## Data split sizes

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| Mine | 141 | 40 | 20 |
| Sanghyeon | 155 | 31 | 15 |
| Jian | 141 | 40 | 20 |
