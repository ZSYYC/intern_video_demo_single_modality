import os
import csv
import random
import sys

# ======================
# 参数设置
# ======================
dataset_root = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/train"
categories_txt = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/kinetics_400_categroies.txt"
output_root = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/output_intern_video_format_dataset"
videos_dir = os.path.join(output_root, "videos")
os.makedirs(videos_dir, exist_ok=True)

label_map_txt = os.path.join(output_root, "label_map.txt")

# 自动切分比例
val_ratio = 0.1   # 10% 做 validation
test_ratio = 0.1  # 10% 做 test
random.seed(42)

# ======================
# 读取类别映射表
# ======================
if not os.path.exists(categories_txt):
    print(f"Error: 类别映射表 {categories_txt} 不存在!")
    sys.exit(1)

with open(categories_txt, "r", encoding="utf-8") as f:
    categories = []
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) < 2:
            print(f"❌ 文件格式错误：{line}")
            sys.exit(1)
        cls_name = parts[0].strip()   # 类别名
        categories.append(cls_name)

class_to_idx = {cls_name: idx for idx, cls_name in enumerate(categories)}

# ======================
# 校验类别并写 label_map
# ======================
dirs = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
for d in dirs:
    if d not in class_to_idx:
        print(f"❌ 错误：类别 {d} 不存在于 {categories_txt} 中！")
        sys.exit(1)

with open(label_map_txt, "w", encoding="utf-8") as f:
    for cls_name, idx in class_to_idx.items():
        f.write(f"{cls_name}\t{idx}\n")

print(f"✅ 类别校验通过，label_map.txt 已生成")

# ======================
# 写 train/val/test.csv
# ======================
splits = {
    "train": open(os.path.join(output_root, "train.csv"), "w", newline='', encoding="utf-8"),
    "val": open(os.path.join(output_root, "val.csv"), "w", newline='', encoding="utf-8"),
    "test": open(os.path.join(output_root, "test.csv"), "w", newline='', encoding="utf-8")
}
writers = {k: csv.writer(v, delimiter=",") for k, v in splits.items()}

for cls_name in categories:
    cls_dir = os.path.join(dataset_root, cls_name)
    if not os.path.exists(cls_dir):
        continue  # 如果类别目录不存在，跳过（但前面已经校验过，几乎不会发生）

    video_files = [vf for vf in os.listdir(cls_dir)
                   if os.path.isfile(os.path.join(cls_dir, vf)) and vf.lower().endswith(".mp4")]
    random.shuffle(video_files)

    n_total = len(video_files)
    if n_total == 0:
        print(f"⚠️ 类别 {cls_name} 下没有 mp4 视频，跳过")
        continue

    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    val_videos = video_files[:n_val]
    test_videos = video_files[n_val:n_val + n_test]
    train_videos = video_files[n_val + n_test:]

    for split_name, split_videos in zip(["val", "test", "train"], [val_videos, test_videos, train_videos]):
        for video_file in split_videos:
            src_path = os.path.join(cls_dir, video_file)
            dst_path = os.path.join(videos_dir, video_file)
            if not os.path.exists(dst_path):
                try:
                    os.symlink(src_path, dst_path)
                except FileExistsError:
                    pass
            writers[split_name].writerow([video_file, class_to_idx[cls_name]])

# 关闭文件
for v in splits.values():
    v.close()

print(f"🎉 处理完成！train/val/test.csv 和 label_map.txt 已生成，视频已拷贝到 {videos_dir}")
