import os
import shutil
import csv

# 输入输出路径
dataset_root = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/train"
output_root = "/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/output_intern_video_format_dataset"
videos_dir = os.path.join(output_root, "videos")
os.makedirs(videos_dir, exist_ok=True)

train_csv = os.path.join(output_root, "train.csv")
label_map_txt = os.path.join(output_root, "label_map_train.txt")

# 获取类别名称并分配索引
classes = sorted([d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))])
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

# 写标签映射文件
with open(label_map_txt, "w", encoding="utf-8") as f:
    for cls_name, idx in class_to_idx.items():
        f.write(f"{cls_name}\t{idx}\n")

# 写train.csv并复制视频
with open(train_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter=",")
    for cls_name, idx in class_to_idx.items():
        cls_dir = os.path.join(dataset_root, cls_name)
        for video_file in os.listdir(cls_dir):
            src_path = os.path.join(cls_dir, video_file)
            if not os.path.isfile(src_path):
                continue
            if not video_file.lower().endswith(".mp4"):
                continue
            # 统一放到videos目录
            dst_path = os.path.join(videos_dir, video_file)
            # 直接复制视频文件
            # shutil.copy2(src_path, dst_path)
            # 使用软链接的方式
            if not os.path.exists(dst_path):
                try:
                    os.symlink(src_path, dst_path)
                except FileExistsError:
                    pass

            # 写入csv文件
            writer.writerow([video_file, idx])

print(f"处理完成！train.csv 和 label_map.txt 已生成，视频已拷贝到 {videos_dir}")
