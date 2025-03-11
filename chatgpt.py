#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author:chomes
# @email:chomeso3o@hotmail.com
# @time:2025/03/11
# @file:chatgpt.py
import os
from openai import OpenAI
import json
import subprocess
import whisper

from check import check_json
from conf import Config
from utils import get_video_length

class Chat:
    def __init__(
            self, api_key=Config.api_key, base_url=Config.base_url, model=Config.model
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, srt_path, video_path, style, depth=1):
        max_depth = 10
        if depth > max_depth:
            print(f"递归达到最大深度 {max_depth}，生成失败，返回空结果")
            return "[]"  # 返回空 JSON，继续处理后续视频

        video_duration_formatted = get_video_length(video_path)
        print(f"视频总时长: {video_duration_formatted}")

        # 读取生成的字幕文件
        if not os.path.exists(srt_path):
            print(f"字幕文件 {srt_path} 不存在，尝试从视频解析")
            srt_path = extract_subtitles(video_path, srt_path)
            if not srt_path:
                print("字幕解析失败，返回空结果")
                return "[]"

        with open(srt_path, "r", encoding="utf-8") as f:
            subtitle_content = f.read()
        print(f"读取的字幕内容:\n{subtitle_content}")

        with open("init_prompt.txt", "r", encoding="utf-8") as f1:
            messages = [
                {
                    "role": "system",
                    "content": f1.read()
                               + "\n"
                               + "## 视频总长度"
                               + "\n"
                               + video_duration_formatted
                               + "\n"
                               + "## 本集字幕内容"
                               + "\n"
                               + subtitle_content
                               + "\n"
                               + "## 风格",
                }
            ]
        msg = messages

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=msg,
                temperature=0.8,
            )
            result = completion.choices[0].message.content
            result = result.replace("```json", "").replace("```", "")

            print(f"生成的JSON内容: {result}")
            try:
                parsed_result = json.loads(result)
                print(f"片段数量: {len(parsed_result)}")
            except json.JSONDecodeError:
                print("生成内容不是有效的JSON格式")

            if check_json(result, video_duration_formatted):
                return result
            else:
                print(f"第{depth}次生成失败，重新尝试...")
                return self.chat(srt_path, video_path, style, depth + 1)
        except Exception as e:
            print(f"生成过程中发生错误: {e}")
            print(f"第{depth}次生成失败，重新尝试...")
            return self.chat(srt_path, video_path, style, depth + 1)

def extract_subtitles(video_path, output_srt_path):
    audio_path = "temp_audio.mp3"
    ffmpeg_path = r"D:\ffmpeg\bin\ffmpeg.exe"
    command = [
        ffmpeg_path, "-i", video_path, "-vn", "-acodec", "mp3", audio_path, "-y"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"提取音频失败: {result.stderr}")
        return None

    try:
        model = whisper.load_model("medium")
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