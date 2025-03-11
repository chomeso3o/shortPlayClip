import asyncio
import os.path
import re
import aiofiles
import edge_tts
from mutagen.mp3 import MP3
import aiohttp
import time

from conf import Config
# @author:chomes
# @email:chomeso3o@hotmail.com
# @time:2025/03/11
async def create_voice_srt_new2(
        index,
        file_txt,
        save_dir,
        time_range,  # JSON 中的绝对时间范围，仅用于参考
        p_voice=Config.voice,
        p_rate=Config.rate,
        p_volume=Config.volume,
        max_retries=3,  # 新增：最大重试次数
        retry_delay=5   # 新增：重试间隔（秒）
):
    mp3_name = f"{index}.mp3"
    srt_name = f"{index}.srt"
    file_mp3 = os.path.join(save_dir, mp3_name)
    file_srt = os.path.join(save_dir, srt_name)

    # 生成音频，重试机制
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(
                text=file_txt, voice=p_voice, rate=p_rate, volume=p_volume
            )
            async with aiofiles.open(file_mp3, "wb") as file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        await file.write(chunk["data"])
            break  # 成功则退出重试
        except aiohttp.client_exceptions.WSServerHandshakeError as e:
            print(f"尝试 {attempt + 1}/{max_retries} 失败: Edge TTS 服务不可用 - {e}")
            if os.path.exists(file_mp3):
                os.remove(file_mp3)
                print(f"File {file_mp3} has been deleted.")
            if attempt < max_retries - 1:
                print(f"将在 {retry_delay} 秒后重试...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"达到最大重试次数 {max_retries}，跳过此片段")
                # 返回默认值，避免程序崩溃
                with open(file_srt, "w", encoding="utf-8") as f:
                    f.write(f"1\n00:00:00,000 --> 00:00:01,000\n{file_txt} (语音生成失败)\n\n")
                return file_mp3, file_srt
        except Exception as e:
            print(f"发生未知错误: {e}")
            if os.path.exists(file_mp3):
                os.remove(file_mp3)
                print(f"File {file_mp3} has been deleted.")
            raise e  # 其他异常仍抛出，供上层捕获

    # 获取音频时长
    try:
        audio = MP3(file_mp3)
        duration_seconds = audio.info.length
        start_time = "00:00:00,000"
        end_time = f"{int(duration_seconds // 3600):02d}:{int((duration_seconds % 3600) // 60):02d}:{int(duration_seconds % 60):02d},{int((duration_seconds % 1) * 1000):03d}"
    except Exception as e:
        print(f"获取音频时长失败: {e}")
        # 默认 1 秒时长
        start_time = "00:00:00,000"
        end_time = "00:00:01,000"

    # 使用相对时间写入 .srt 文件
    with open(file_srt, "w", encoding="utf-8") as f:
        f.write(f"1\n{start_time} --> {end_time}\n{file_txt}\n\n")

    # 调试输出
    with open(file_srt, "r", encoding="utf-8") as f:
        print(f"{srt_name} 内容：\n{f.read()}")

    return file_mp3, file_srt

# 其他函数保持不变（如 spilt_str2, load_srt_new 等）

if __name__ == "__main__":
    file_name = "测试"
    text = "今天我们带来了一部充满正能量的影视剧"
    save_path = "./"
    asyncio.run(create_voice_srt_new2(file_name, text, save_path))