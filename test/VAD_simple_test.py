#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import sys
import os
import time
import threading
from colorama import init, Fore, Back, Style

# 初始化 Colorama
init(autoreset=True)

def clear_line():
    """清除当前行"""
    print('\r' + ' ' * 80 + '\r', end='')

def print_header():
    """打印标题"""
    clear_line()
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                           FunASR VAD 简化测试程序{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}正在测试色彩显示和系统状态...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")

def test_funasr():
    """测试 FunASR 加载"""
    print(f"{Fore.YELLOW}[1/4] 正在测试 FunASR 模块加载...{Style.RESET_ALL}")
    try:
        from funasr import AutoModel
        print(f"{Fore.GREEN}✓ FunASR 模块加载成功{Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}[2/4] 正在加载语音活动检测模型...{Style.RESET_ALL}")
        # 设置模型缓存
        current_dir = os.getcwd()
        os.environ['MODELSCOPE_CACHE'] = current_dir
        os.environ['HF_HOME'] = current_dir
        
        model = AutoModel(
            model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
            model_revision="v2.0.4"
        )
        print(f"{Fore.GREEN}✓ VAD 模型加载成功{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}✗ FunASR 加载失败: {str(e)}{Style.RESET_ALL}")
        return False

def test_audio():
    """测试音频设备"""
    print(f"{Fore.YELLOW}[3/4] 正在测试音频设备...{Style.RESET_ALL}")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        # 列出音频设备
        device_count = p.get_device_count()
        print(f"{Fore.CYAN}发现 {device_count} 个音频设备:{Style.RESET_ALL}")
        
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {device_info['name']} (输入通道: {device_info['maxInputChannels']})")
        
        p.terminate()
        print(f"{Fore.GREEN}✓ 音频设备检测完成{Style.RESET_ALL}")
        return True
    except Exception as e:
        print(f"{Fore.RED}✗ 音频设备测试失败: {str(e)}{Style.RESET_ALL}")
        return False

def test_colorama():
    """测试 Colorama 色彩"""
    print(f"{Fore.YELLOW}[4/4] 正在测试色彩显示...{Style.RESET_ALL}")
    
    colors = [
        (Fore.RED, "红色"),
        (Fore.GREEN, "绿色"),
        (Fore.YELLOW, "黄色"),
        (Fore.BLUE, "蓝色"),
        (Fore.MAGENTA, "洋红色"),
        (Fore.CYAN, "青色"),
        (Fore.WHITE, "白色")
    ]
    
    for color, name in colors:
        print(f"  {color}● {name} 测试{Style.RESET_ALL}")
    
    print(f"  {Style.BRIGHT}粗体文本测试{Style.RESET_ALL}")
    print(f"  {Style.DIM}暗淡文本测试{Style.RESET_ALL}")
    print(f"  {Back.RED}{Fore.WHITE}背景色测试{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✓ 色彩显示测试完成{Style.RESET_ALL}")
    return True

def run_test():
    """运行所有测试"""
    print_header()
    
    results = []
    
    # 测试各个组件
    results.append(test_funasr())
    results.append(test_audio())
    results.append(test_colorama())
    
    # 显示结果
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                             测试结果汇总{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    
    if all(results):
        print(f"{Fore.GREEN}{Style.BRIGHT}🎉 所有测试通过！系统运行正常{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}可以开始使用 VAD 录音系统{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}{Style.BRIGHT}❌ 部分测试失败{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}请检查上述错误信息{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")

def main():
    """主函数"""
    try:
        run_test()
        
        # 等待用户输入
        print(f"{Fore.CYAN}按 Enter 键退出...{Style.RESET_ALL}", end='')
        input()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}用户中断程序{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}程序发生错误: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
