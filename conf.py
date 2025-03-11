#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author:chomes
# @email:chomeso3o@hotmail.com
# @time:2025/03/11
# @file:conf.py


class Config:
    # 单个视频路径
    video_path = r"resource/video/"
    # 单个srt路径
    srt_path = r""
    # ChatGPT API key 支持本地 和各类国内代理商
    api_key = ""
    #  ChatGPT API 请求路由
    base_url = ""
    # 模型
    model = "gpt-4o-2024-11-20"
    # 风格列表
    style_list = [

        "搞笑和讽刺风格：通过搞笑和夸大或者讽刺、一阵见血的手法来评论剧中的不合理或过于狗血的情节，让观众在笑声中进行思考。",

    ]
    # 声音
    voice = "zh-CN-YunxiNeural"
    # 语速
    rate = "+30%"
    # 音量
    volume = "+100%"
    # 蒙版高度
    blur_height = 185
    # 蒙版位置
    blur_y = 1413
    # 字幕位置
    MarginV = 65
    # 粒子特效目录
    lz_path = None
