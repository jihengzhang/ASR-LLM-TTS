#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
from colorama import init, Fore, Back, Style

# 初始化 Colorama
init(autoreset=True)

def print_header():
    """打印标题"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}                           色彩显示测试程序{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}正在测试 Windows 终端色彩显示...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")

def test_colorama():
    """测试 Colorama 色彩"""
    print(f"{Fore.YELLOW}[1/3] 正在测试基本色彩...{Style.RESET_ALL}")
    
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
        print(f"  {color}● {name} 测试 - 如果您看到这是{name}的，说明色彩正常{Style.RESET_ALL}")
        time.sleep(0.2)
    
    print(f"{Fore.GREEN}✓ 基本色彩测试完成{Style.RESET_ALL}\n")

def test_styles():
    """测试样式"""
    print(f"{Fore.YELLOW}[2/3] 正在测试文本样式...{Style.RESET_ALL}")
    
    print(f"  {Style.BRIGHT}这是粗体文本{Style.RESET_ALL}")
    print(f"  {Style.DIM}这是暗淡文本{Style.RESET_ALL}")
    print(f"  {Back.RED}{Fore.WHITE} 这是红色背景白色文字 {Style.RESET_ALL}")
    print(f"  {Back.GREEN}{Fore.BLACK} 这是绿色背景黑色文字 {Style.RESET_ALL}")
    print(f"  {Back.BLUE}{Fore.YELLOW} 这是蓝色背景黄色文字 {Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✓ 文本样式测试完成{Style.RESET_ALL}\n")

def test_audio_basic():
    """测试基本音频功能"""
    print(f"{Fore.YELLOW}[3/3] 正在测试音频模块...{Style.RESET_ALL}")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        device_count = p.get_device_count()
        print(f"{Fore.CYAN}发现 {device_count} 个音频设备:{Style.RESET_ALL}")
        
        input_devices = []
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
                print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {device_info['name']} (输入通道: {device_info['maxInputChannels']})")
        
        p.terminate()
        
        if input_devices:
            print(f"{Fore.GREEN}✓ 找到 {len(input_devices)} 个可用的音频输入设备{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ 没有找到可用的音频输入设备{Style.RESET_ALL}")
        
        return True
    except Exception as e:
        print(f"{Fore.RED}✗ 音频设备测试失败: {str(e)}{Style.RESET_ALL}")
        return False

def show_progress():
    """显示进度条演示"""
    print(f"\n{Fore.CYAN}进度条演示:{Style.RESET_ALL}")
    for i in range(11):
        progress = '█' * i + '░' * (10 - i)
        percentage = i * 10
        print(f"\r  {Fore.GREEN}[{progress}]{Style.RESET_ALL} {percentage}%", end='')
        time.sleep(0.3)
    print(f"\n{Fore.GREEN}✓ 进度条演示完成{Style.RESET_ALL}\n")

def main():
    """主函数"""
    try:
        print_header()
        
        # 测试各个组件
        test_colorama()
        test_styles()
        audio_ok = test_audio_basic()
        show_progress()
        
        # 显示结果
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}                             测试结果汇总{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}✓ 色彩显示: 正常{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ 文本样式: 正常{Style.RESET_ALL}")
        
        if audio_ok:
            print(f"{Fore.GREEN}✓ 音频设备: 正常{Style.RESET_ALL}")
            print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 所有基础测试通过！{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}现在可以尝试运行完整的 VAD 程序{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠ 音频设备: 有问题{Style.RESET_ALL}")
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}⚠ 音频设备有问题，请检查麦克风连接{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}\n")
        
        # 等待用户输入
        print(f"{Fore.CYAN}按 Enter 键退出...{Style.RESET_ALL}", end='')
        input()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}用户中断程序{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}程序发生错误: {str(e)}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
