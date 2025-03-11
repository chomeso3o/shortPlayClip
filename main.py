#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author:chomes
# @email:chomeso3o@hotmail.com
# @time:2025/03/11
import asyncio
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import requests
from mutagen.mp3 import MP3
from random import sample
import whisper
import torch

from char2voice import create_voice_srt_new2
from chatgpt import Chat
from conf import Config
from utils import get_video_length

time_pattern = re.compile(r"^\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}$")

def compare_time_strings(time1, time2):
    time_format = "%H:%M:%S,%f"
    t1 = datetime.strptime(time1, time_format)
    t2 = datetime.strptime(time2, time_format)
    return t1 >= t2

def is_valid_time(video_duration_formatted, time_str):
    if not time_pattern.match(time_str):
        return False
    start_time, end_time = time_str.split(" --> ")
    return all(compare_time_strings(video_duration_formatted, t) for t in [start_time, end_time]) and compare_time_strings(end_time, start_time)

def check_json(content, video_duration_formatted):
    try:
        data = json.loads(content)
        previous_end_time = "00:00:00,000"
        for i, item in enumerate(data):
            if "type" not in item or "time" not in item:
                print(f"文件错误，缺少必要的字段 at index {i}")
                return False
            if item["type"] not in ["解说", "video"]:
                print(f"文件错误，type字段只能是'解说'或'video' at index {i}")
                return False
            if not is_valid_time(video_duration_formatted, item["time"]):
                print(f"文件错误，时间格式不正确 at index {i}: {item['time']}")
                return False
            start_time, end_time = item["time"].split(" --> ")
            if not compare_time_strings(start_time, previous_end_time):
                print(f"文件错误，下一段的开始时间必须大于或等于上一段的结束时间 at index {i}: {start_time} < {previous_end_time}")
                return False
            previous_end_time = end_time
            if item["type"] == "解说" and "content" not in item:
                print(f"文件错误，缺少content字段 at index {i}")
                return False
        print("JSON校验通过")
        return True
    except json.JSONDecodeError:
        print("文件错误，内容不是有效的JSON格式")
        return False

def extract_subtitles(video_path, output_srt_path):
    audio_path = "temp_audio.mp3"
    ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"
    command = [ffmpeg_path, "-i", video_path, "-vn", "-acodec", "mp3", audio_path, "-y"]
    result = subprocess.run(command, capture_output=True, text=False)
    if result.returncode != 0:
        print(f"提取音频失败: {result.stderr.decode('utf-8', errors='ignore')}")
        return None

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        model = whisper.load_model("small").to(device)
        result = model.transcribe(audio_path, language="zh")
    except Exception as e:
        print(f"Whisper 转录失败: {e}")
        return None

    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]
            start_time = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{int(start % 60):02d},{int((start % 1) * 1000):03d}"
            end_time = f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{int(end % 60):02d},{int((end % 1) * 1000):03d}"
            f.write(f"{i}\n{start_time} --> {end_time}\n{text}\n\n")

    os.remove(audio_path)
    print(f"生成的字幕文件: {output_srt_path}")
    return output_srt_path


class Playlet:
    def sort_by_number(self, filename):
        numbers = re.findall(r"\d+", filename)
        return [int(num) for num in numbers] if numbers else filename

    def run(self):
        video_folder = os.path.dirname(Config.video_path)
        print(video_folder)
        output_base = os.path.join(video_folder, "output")
        os.makedirs(output_base, exist_ok=True)

        drama_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f)) and "-" in f]
        print(f"发现 {len(drama_folders)} 个短剧文件夹: {drama_folders}")

        for drama_folder in drama_folders:
            match = re.match(r"(\d+)-(.+)", drama_folder)
            if not match:
                print(f"跳过无效文件夹: {drama_folder}")
                continue
            drama_id, drama_name = match.groups()

            drama_path = os.path.join(video_folder, drama_folder)
            output_mp4_folder = os.path.join(output_base, drama_name, "mp4")
            output_txt_folder = os.path.join(output_base, drama_name, "txt")
            output_srt_folder = os.path.join(output_base, drama_name, "srt")
            os.makedirs(output_mp4_folder, exist_ok=True)
            os.makedirs(output_txt_folder, exist_ok=True)
            os.makedirs(output_srt_folder, exist_ok=True)

            video_files = [f for f in os.listdir(drama_path) if f.endswith(".mp4")]
            video_files.sort(key=self.sort_by_number)
            print(f"\n处理短剧: {drama_name}，发现 {len(video_files)} 个剧集: {video_files}")

            for video_file in video_files:
                episode_num = re.search(r"(\d+)", video_file).group(1)
                video_path = os.path.join(drama_path, video_file)
                print(f"\n处理剧集: {video_path}")

                try:
                    video_duration = get_video_length(video_path)
                    print(f"视频时长: {video_duration}")
                except Exception as e:
                    print(f"获取视频时长失败: {e}")
                    continue

                for style in Config.style_list:
                    style_prefix = style.split("：")[0]
                    txt_path = os.path.join(output_txt_folder, f"{drama_name}-{episode_num}_{style_prefix}.txt")
                    out_path = os.path.join(output_mp4_folder, f"{drama_name}-{episode_num}_{style_prefix}.mp4")
                    temp_srt_path = os.path.join(output_srt_folder, f"temp_subtitles_{drama_name}-{episode_num}.srt")

                    if not os.path.exists(temp_srt_path):
                        extract_subtitles(video_path, temp_srt_path)

                    if not os.path.exists(txt_path):
                        result = Chat().chat(temp_srt_path, video_path, style)
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(result)
                    else:
                        with open(txt_path, "r", encoding="utf-8") as f:
                            result = f.read()

                    if not check_json(result, video_duration):
                        print(f"Skipping {txt_path} due to invalid JSON")
                        continue

                    data = json.loads(result)
                    full_segments = []
                    video_start = "00:00:00,000"
                    video_end = video_duration
                    for item in data:
                        if item["type"] == "解说":
                            start_time, end_time = item["time"].split(" --> ")
                            if video_start != start_time:
                                full_segments.append({"type": "video", "time": f"{video_start} --> {start_time}"})
                            full_segments.append(item)
                            video_start = end_time
                    if video_start != video_end:
                        full_segments.append({"type": "video", "time": f"{video_start} --> {video_end}"})

                    for k, v in enumerate(full_segments):
                        segment_mp4 = os.path.join(output_mp4_folder, f"{drama_name}-{episode_num}_{style_prefix}_{k}.mp4")
                        if os.path.exists(segment_mp4) and os.path.getsize(segment_mp4) > 0:
                            print(f"{segment_mp4} already exists and valid, skipping")
                            continue

                        start_time = v["time"].split(" --> ")[0]
                        if v["type"] == "解说":
                            mp3_path = os.path.join(output_mp4_folder, f"{drama_name}-{episode_num}_{style_prefix}_{k}.mp3")
                            srt_path = os.path.join(output_srt_folder, f"{drama_name}-{episode_num}_{style_prefix}_{k}.srt")
                            duration = self.generate_speech(v["content"], mp3_path, srt_path, v["time"])
                            end_time = self.add_seconds_to_time(start_time, duration)
                        # 在处理"video"类型片段时（当前这部分代码可能没实现或实现有问题）
                        # ... existing code ...
                        else:  # v["type"] == "video"
                            end_time = v["time"].split(" --> ")[-1]
                            duration_str = self.calculate_time_difference_srt(v["time"])

                            # 将字符串时间转换为浮点数秒数
                            start_time_for_trim = v["time"].split(" --> ")[0].replace(',', '.')

                            # 简单方案：直接使用原始格式
                            duration_for_trim = duration_str.replace(',', '.')

                            if not os.path.exists(segment_mp4):
                                print(f"Generating {segment_mp4} with start_time={start_time_for_trim}, duration={duration_for_trim}")
                                self.trim_video(
                                    video_path, segment_mp4,
                                    start_time_for_trim, duration_for_trim,
                                    Config.lz_path,
                                    border_width=20, border_color="yellow",
                                    top_black=200, bottom_black=200,
                                    drama_name=drama_name
                                )
                            # ... existing code ...



                        print(f"Segment {k} start_time: {start_time}, duration: {duration}, end_time: {end_time}")

                        if duration.startswith("-") or duration == "00:00:00.000":
                            print(f"Invalid duration for segment {k}, skipping")
                            continue

                        start_time_for_trim = start_time.replace(",", ".")
                        duration_for_trim = duration.replace(",", ".")
                        if not os.path.exists(segment_mp4):
                            print(f"Generating {segment_mp4} with start_time={start_time_for_trim}, duration={duration_for_trim}")
                            self.trim_video(
                                video_path, segment_mp4,
                                start_time_for_trim, duration_for_trim,
                                Config.lz_path,
                                border_width=20, border_color="yellow",
                                top_black=200, bottom_black=200
                            )
                            if not os.path.exists(segment_mp4) or os.path.getsize(segment_mp4) == 0:
                                print(f"Failed to generate {segment_mp4}")
                                continue

                        if v["type"] == "解说":
                            self.process_video(
                                segment_mp4, mp3_path, srt_path, segment_mp4,
                                drama_name,
                                border_width=20, border_color="black",
                                top_black=100, bottom_black=100
                            )


                    video_files_to_concat = [
                        os.path.join(output_mp4_folder, f"{drama_name}-{episode_num}_{style_prefix}_{i_}.mp4")
                        for i_, v in enumerate(full_segments)
                        if os.path.exists(os.path.join(output_mp4_folder, f"{drama_name}-{episode_num}_{style_prefix}_{i_}.mp4"))
                           and os.path.getsize(os.path.join(output_mp4_folder, f"{drama_name}-{episode_num}_{style_prefix}_{i_}.mp4")) > 0
                    ]
                    print(f"Files to concatenate: {video_files_to_concat}")
                    if video_files_to_concat:
                        self.concat_videos(video_files_to_concat, out_path)
                    else:
                        print(f"没有可合并的文件，跳过 {out_path}")

    def calculate_time_difference_srt(self, srt_timestamp):
        start_time_str, end_time_str = srt_timestamp.replace(",", ".").split(" --> ")
        time_format = "%H:%M:%S.%f" if "." in start_time_str else "%H:%M:%S"
        start_time = datetime.strptime(start_time_str, time_format)
        end_time = datetime.strptime(end_time_str, time_format)
        time_difference = end_time - start_time
        total_seconds = time_difference.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    def generate_speech(self, text, mp3_path, srt_path, time_range,
                        p_voice=Config.voice,
                        p_rate=Config.rate,
                        p_volume=Config.volume):
        if not os.path.exists(mp3_path):
            print(f"Generating speech for {mp3_path}")
            file_name = os.path.splitext(os.path.basename(mp3_path))[0]
            output_dir = os.path.dirname(mp3_path)
            asyncio.run(create_voice_srt_new2(file_name, text, output_dir, time_range, p_voice, p_rate, p_volume))
            temp_srt = os.path.join(output_dir, f"{file_name}.srt")
            if os.path.exists(temp_srt) and temp_srt != srt_path:
                os.rename(temp_srt, srt_path)
        duration = self.get_mp3_length_formatted(mp3_path)
        with open(srt_path, "r", encoding="utf-8") as f:
            print(f"调整后的 {srt_path} 内容：\n{f.read()}")
        return duration

    def get_mp3_length_formatted(self, file_path):
        audio = MP3(file_path)
        total_seconds = audio.info.length
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        milliseconds = int((total_seconds % 1) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

    def add_seconds_to_time(self, time_str, seconds_to_add):
        try:
            h, m, s = seconds_to_add.split(':')
            if '.' in s:
                s, ms = s.split('.')
                total_seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            else:
                total_seconds = int(h) * 3600 + int(m) * 60 + int(s)
            time_str = time_str.replace(',', '.')
            time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
            new_time_obj = time_obj + timedelta(seconds=total_seconds)
            new_time_str = new_time_obj.strftime("%H:%M:%S,%f")[:-3]
            return new_time_str
        except ValueError as e:
            print(f"Error in add_seconds_to_time: {e}")
            return time_str
    def trim_video(self, input_path, output_path, start_time, duration, lz_path=None, log_level="info",
                   border_width=0, border_color="black", top_black=0, bottom_black=0, drama_name=None):
        """裁剪视频并添加文字提示"""
        ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"

        # 基本的裁剪命令
        filter_str = f"pad=iw:ih+{top_black}+{bottom_black}:0:{top_black},setsar=1:1"

        # 如果传入了剧名参数，添加文字
        if drama_name:
            # 使用绝对路径指定字体文件，避免fontconfig错误
            font_path = "C\\:/Windows/Fonts/simhei.ttf"

            # 添加文字绘制滤镜
            filter_str += (
                f",drawtext=fontfile='{font_path}':text='{drama_name}':fontcolor=white:fontsize=50:"
                f"x=(w-text_w)/2:y=10"
            )

            # 添加"剧情内容请勿模仿"文字
            filter_str += (
                f",drawtext=fontfile='{font_path}':text='剧情内容请勿模仿':fontcolor=red:fontsize=70:"
                f"x=(w-text_w)/2:y=h-text_h-60"
            )

            # 添加左侧"完整版点击左下角"
            filter_str += (
                f",drawtext=fontfile='{font_path}':text='完整版点击左下角':fontcolor=white:fontsize=70:"
                f"x=20:y=(h-text_h)/2"
            )

            # 添加右侧"树立正确价值观"
            filter_str += (
                f",drawtext=fontfile='{font_path}':text='树立正确价值观':fontcolor=white:fontsize=70:"
                f"x=w-text_w-20:y=(h-text_h)/2"
            )

        command = [
            ffmpeg_path, "-v", log_level, "-i", input_path,
            "-ss", start_time, "-t", duration,
            "-vf", filter_str,
            "-c:v", "libx264", "-b:v", "2000k", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k", "-ar", "24000", "-ac", "2",
            output_path, "-y"
        ]

        result = subprocess.run(command, capture_output=True, text=False)
        if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Error trimming video {output_path}: {result.stderr.decode('utf-8', errors='ignore')}")
        else:
            print(f"Successfully generated {output_path}, 文件大小: {os.path.getsize(output_path)} 字节")


    def process_video(self, video_path, audio_path, subtitle_path, output_path, drama_name,
                      blur_height=Config.blur_height, blur_y=Config.blur_y,
                      MarginV=Config.MarginV, log_level="info",
                      border_width=20, border_color="black",
                      top_black=100, bottom_black=100):
        """
        在该函数中，对每帧执行：
          1. 模糊指定区域
          2. 添加黑边
          3. 添加边框
          4. 顶部剧名
          5. (可选) 添加字幕
          6. 底部“剧情内容请勿模仿”
          7. 左侧竖排“完整版点击左下角”，竖排+居中
          8. 右侧竖排“树立正确价值观”，竖排+居中
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_output = output_path + ".tmp.mp4"
        out = cv2.VideoWriter(temp_video_output, fourcc, fps,
                              (frame_width, frame_height + top_black + bottom_black))

        # 加载字体（根据需要调整）
        font_path = "C:/Windows/Fonts/simhei.ttf"
        title_font = ImageFont.truetype(font_path, 50) if os.path.exists(font_path) else ImageFont.load_default()   # 剧名字体
        subtitle_font = ImageFont.truetype(font_path, 70) if os.path.exists(font_path) else ImageFont.load_default() # 字幕字体
        tips_font = ImageFont.truetype(font_path, 70) if os.path.exists(font_path) else ImageFont.load_default()    # “剧情内容请勿模仿”等提示文字

        # 读取字幕
        subtitles = self.parse_srt(subtitle_path)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 模糊指定区域
            if blur_y < frame_height and blur_height > 0:
                blur_region = frame[blur_y:blur_y + blur_height, :]
                blurred = cv2.GaussianBlur(blur_region, (21, 21), 20)
                frame[blur_y:blur_y + blur_height, :] = blurred

            # 2. 添加黑边（上下）
            frame_with_borders = cv2.copyMakeBorder(
                frame, top_black, bottom_black, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            # 3. 添加边框
            cv2.rectangle(
                frame_with_borders,
                (0, 0),
                (frame_width - 1, frame_height + top_black + bottom_black - 1),
                (0, 0, 0) if border_color == "black" else (0, 255, 255),
                border_width
            )

            # 4. 顶部剧名（水平居中）
            frame_with_borders = self.add_text_to_frame(
                frame_with_borders, drama_name, title_font,
                y_offset=10, text_color=(255, 255, 255), bottom=False
            )

            # 5. 如果需要显示自动识别的字幕（底部水平居中）
            current_time = frame_idx / fps
            subtitle_text = self.get_subtitle_at_time(subtitles, current_time)
            if subtitle_text:
                frame_with_borders = self.add_text_to_frame(
                    frame_with_borders,
                    subtitle_text,
                    subtitle_font,
                    y_offset=MarginV,  # 距离底部的距离
                    text_color=(255, 255, 255),
                    bottom=True
                )

            # 6. 底部“剧情内容请勿模仿”
            frame_with_borders = self.add_text_to_frame(
                frame_with_borders,
                "剧情内容请勿模仿",
                tips_font,
                y_offset=MarginV - 60,  # 比字幕再往上一点，防止重叠
                text_color=(255, 0, 0),
                bottom=True
            )

            # 7. 左侧竖排“完整版点击左下角”
            #    居中 => y 在画面的垂直中点 - 总高度/2
            frame_with_borders = self.add_vertical_text_line_by_line(
                frame_with_borders,
                "完整版点击左下角",
                tips_font,
                position='left',               # 靠左
                text_color=(255, 255, 255),
                border_offset=20               # 距离左侧边缘多少像素
            )

            # 8. 右侧竖排“树立正确价值观”
            frame_with_borders = self.add_vertical_text_line_by_line(
                frame_with_borders,
                "树立正确价值观",
                tips_font,
                position='right',              # 靠右
                text_color=(255, 255, 255),
                border_offset=20               # 距离右侧边缘多少像素
            )

            out.write(frame_with_borders)
            frame_idx += 1

        cap.release()
        out.release()

        # 使用 FFmpeg 替换音频
        ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"
        final_command = [
            ffmpeg_path, "-v", log_level, "-y", "-i", temp_video_output, "-i", audio_path,
            "-c:v", "libx264", "-b:v", "2000k", "-preset", "fast",
            "-c:a", "aac", "-b:a", "128k", "-ar", "24000", "-ac", "2",
            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
            output_path
        ]
        result = subprocess.run(final_command, capture_output=True, text=False)
        if result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"Error processing video {output_path}: {result.stderr.decode('utf-8', errors='ignore')}")
            if os.path.exists(temp_video_output):
                os.remove(temp_video_output)
        else:
            os.remove(temp_video_output)
            os.remove(subtitle_path)
            os.remove(audio_path)
            print(f"Successfully processed {output_path}, 文件大小: {os.path.getsize(output_path)} 字节")

    def parse_srt(self, srt_path):
        """解析 SRT 文件"""
        subtitles = []
        if not os.path.exists(srt_path):
            print(f"字幕文件 {srt_path} 不存在")
            return subtitles
        with open(srt_path, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")
            for entry in blocks:
                lines = entry.split("\n")
                if len(lines) >= 3:
                    time_line = lines[1]
                    start_time, end_time = time_line.split(" --> ")
                    text = "\n".join(lines[2:])
                    subtitles.append({
                        "start": self.time_to_seconds(start_time),
                        "end": self.time_to_seconds(end_time),
                        "text": text
                    })
        return subtitles

    def get_subtitle_at_time(self, subtitles, current_time):
        """获取当前时间的字幕文本"""
        for sub in subtitles:
            if sub["start"] <= current_time <= sub["end"]:
                return sub["text"]
        return None

    def add_text_to_frame(self, frame, text, font, y_offset,
                          text_color=(255, 255, 255), bottom=False):
        """
        在帧上绘制文字（水平排版）：
          - bottom=True 表示贴近底部
          - bottom=False 表示贴近顶部
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 计算文字大小
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 居中计算
        img_w, img_h = pil_image.size
        x = (img_w - text_w) // 2
        if bottom:
            y = img_h - y_offset - text_h
        else:
            y = y_offset

        draw.text((x, y), text, font=font, fill=text_color)

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def add_vertical_text_line_by_line(self, frame, text, font,
                                       position='left',
                                       text_color=(255, 255, 255),
                                       border_offset=20):
        """
        逐字竖排绘制，让文字能正常阅读：
          - position='left' 或 'right' 决定横向位置
          - 在垂直方向上居中，让整列文字在画面中部
        """
        if not text:
            return frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        # 将每个字符拆分成单独行
        # 例如 "完整版点击左下角" => ["完","整","版","点","击","左","下","角"]
        chars = list(text.strip())

        # 先测量每个字符的高度，并统计总高度
        line_heights = []
        for ch in chars:
            bbox = draw.textbbox((0, 0), ch, font=font)
            ch_h = bbox[3] - bbox[1]
            line_heights.append(ch_h)

        total_h = sum(line_heights)

        # 画布大小
        img_w, img_h = pil_image.size

        # x 坐标
        #  如果靠左 => x = border_offset
        #  如果靠右 => x = img_w - border_offset - （字符最大宽度）
        max_ch_width = max(
            (draw.textbbox((0, 0), ch, font=font)[2] -
             draw.textbbox((0, 0), ch, font=font)[0])
            for ch in chars
        )

        if position == 'left':
            x = border_offset
        else:  # 'right'
            x = img_w - border_offset - max_ch_width

        # y 坐标（让整列文字居中 => (img_h - total_h)/2）
        y = (img_h - total_h) // 2

        # 逐字写入
        for i, ch in enumerate(chars):
            draw.text((x, y), ch, font=font, fill=text_color)
            y += line_heights[i]  # 下一行往下移

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def concat_videos(self, video_files, output_file, log_level="info"):
        if not video_files:
            print(f"错误：没有可合并的视频文件，跳过 {output_file}")
            return

        with open("filelist.txt", "w", encoding="utf-8") as file:
            for video in video_files:
                abs_video_path = os.path.abspath(video)
                if os.path.exists(abs_video_path):
                    file.write(f"file '{abs_video_path}'\n")
                    print(f"添加合并文件: {abs_video_path}, 大小: {os.path.getsize(abs_video_path)} 字节")
                else:
                    print(f"警告：文件 {abs_video_path} 不存在，跳过此文件")

        if not os.path.exists("filelist.txt") or os.path.getsize("filelist.txt") == 0:
            print(f"错误：filelist.txt 未生成或为空，无法合并 {output_file}")
            return

        ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"
        command = [
            ffmpeg_path, "-loglevel", log_level, "-y", "-f", "concat", "-safe", "0",
            "-i", "filelist.txt", "-c", "copy", output_file
        ]
        print(f"执行合并命令: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=False)
        print(f"FFmpeg 输出: {result.stdout.decode('utf-8', errors='ignore')}")
        print(f"FFmpeg 错误: {result.stderr.decode('utf-8', errors='ignore')}")
        if result.returncode != 0:
            print(f"合并视频失败: {result.stderr.decode('utf-8', errors='ignore')}")
        else:
            print(f"Successfully concatenated to {output_file}, 文件大小: {os.path.getsize(output_file)} 字节")
            os.remove("filelist.txt")
            for video in video_files:
                if os.path.exists(video):
                    os.remove(video)

    def get_video(self, path):
        return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name != 'Thumbs.db']

    def time_to_seconds(self, time_str):
        time_str = time_str.replace(",", ".")
        time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
        return (time_obj.hour * 3600 + time_obj.minute * 60 +
                time_obj.second + time_obj.microsecond / 1000000)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        if len(sys.argv) != 3:
            print("Usage: python main.py client <server_url>")
            sys.exit(1)
        server_url = sys.argv[2]
        Playlet().client(server_url)
    else:
        Playlet().run()
