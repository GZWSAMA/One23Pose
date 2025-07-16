import cv2
import os

def extract_frames(video_path, output_folder):
    """
    从视频中提取所有帧，并按顺序保存为 frame_000000.png 格式。

    参数:
        video_path (str): 视频文件的路径。
        output_folder (str): 保存帧图像的输出目录。

    返回:
        int: 提取并保存的总帧数。
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    frame_idx = 0
    file_names = []
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 构造文件名，例如：frame_000000.png
        filename = os.path.join(output_folder, f"{frame_idx:06d}.png")
        file_names.append(filename)

        # 保存帧为图像文件
        cv2.imwrite(filename, frame)

        frame_idx += 1

    # 释放资源
    cap.release()

    print(f"共保存 {frame_idx} 帧图像到 '{output_folder}'。")
    return file_names

