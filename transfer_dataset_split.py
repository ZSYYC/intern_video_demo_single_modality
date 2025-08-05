import os
import shutil
import csv
import random

# ======================
# 参数设置
# ======================
dataset_root = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/train"
output_root = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/output_intern_video_format_dataset"
videos_dir = os.path.join(output_root, "videos")
os.makedirs(videos_dir, exist_ok=True)

label_map_txt = os.path.join(output_root, "label_map_train.txt")

# 自动切分比例
val_ratio = 0.1   # 10% 做 validation
test_ratio = 0.1  # 10% 做 test
# 随机种子保证可复现
random.seed(42)

# ======================
# 获取类别名称并分配索引
# ======================
classes = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# 写标签映射文件
with open(label_map_txt, "w", encoding="utf-8") as f:
    for cls_name, idx in class_to_idx.items():
        f.write(f"{cls_name}\t{idx}\n")

# ======================
# 写 train/val/test.csv 并复制视频
# ======================
splits = {
    "train": open(os.path.join(output_root, "train.csv"), "w", newline='', encoding="utf-8"),
    "val": open(os.path.join(output_root, "val.csv"), "w", newline='', encoding="utf-8"),
    "test": open(os.path.join(output_root, "test.csv"), "w", newline='', encoding="utf-8")
}
writers = {k: csv.writer(v, delimiter=",") for k, v in splits.items()}

for cls_name, idx in class_to_idx.items():
    cls_dir = os.path.join(dataset_root, cls_name)
    video_files = [vf for vf in os.listdir(cls_dir)
                   if os.path.isfile(os.path.join(cls_dir, vf)) and vf.lower().endswith(".mp4")]

    # 随机打乱
    random.shuffle(video_files)

    n_total = len(video_files)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    # 切分
    val_videos = video_files[:n_val]
    test_videos = video_files[n_val:n_val + n_test]
    train_videos = video_files[n_val + n_test:]

    # 写入 csv & 建立软链接
    for split_name, split_videos in zip(["val", "test", "train"], [val_videos, test_videos, train_videos]):
        for video_file in split_videos:
            src_path = os.path.join(cls_dir, video_file)
            dst_path = os.path.join(videos_dir, video_file)
            if not os.path.exists(dst_path):
                try:
                    os.symlink(src_path, dst_path)
                except FileExistsError:
                    pass
            writers[split_name].writerow([video_file, idx])

# 关闭文件
for v in splits.values():
    v.close()

print(f"处理完成！train/val/test.csv 和 label_map.txt 已生成，视频已拷贝到 {videos_dir}")
