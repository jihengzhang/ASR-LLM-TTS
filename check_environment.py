"""
FunASR 环境检查工具
检查 FunASR 运行所需的所有依赖是否已正确安装
"""

import os
import sys
import subprocess
import importlib.util
import platform

def print_status(message, status, error_msg=None):
    """打印状态信息"""
    if status:
        print(f"[✓] {message}")
    else:
        print(f"[✗] {message}: {error_msg or '未安装或配置不正确'}")
    return status

def check_command(command, args=None):
    """检查命令是否可用"""
    if args is None:
        args = ["-version"]
    try:
        subprocess.run([command] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def check_module(module_name):
    """检查 Python 模块是否已安装"""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def get_python_version():
    """获取 Python 版本"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def main():
    print("\n===== FunASR 环境检查工具 =====\n")
    
    # 检查操作系统
    os_name = platform.system()
    os_version = platform.version()
    print(f"操作系统: {os_name} {os_version}")
    
    # 检查 Python 版本
    python_version = get_python_version()
    python_ok = sys.version_info.major == 3 and sys.version_info.minor >= 8
    print_status(f"Python 版本: {python_version}", python_ok, "需要 Python 3.8 或更高版本")
    
    # 检查 ffmpeg
    ffmpeg_ok = check_command("ffmpeg")
    print_status("ffmpeg", ffmpeg_ok)
    
    # 检查必要的 Python 模块
    modules = {
        "numpy": "数值计算库",
        "pyaudio": "音频处理库",
        "scipy": "科学计算库",
        "torch": "PyTorch",
        "torchaudio": "PyTorch 音频处理",
        "funasr": "FunASR 语音识别"
    }
    
    print("\n依赖库检查:")
    all_modules_ok = True
    for module, description in modules.items():
        module_ok = check_module(module)
        if not print_status(f"{module} ({description})", module_ok):
            all_modules_ok = False
    
    # 如果 FunASR 已安装，检查模型是否可用
    if check_module("funasr"):
        try:
            print("\n尝试加载 FunASR VAD 模型...")
            from funasr import AutoModel
            vad_model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4")
            print_status("FunASR VAD 模型加载", True)
        except Exception as e:
            print_status("FunASR VAD 模型加载", False, str(e))
    
    # 总结
    print("\n===== 检查结果 =====")
    if not ffmpeg_ok:
        print("[!] 缺少 ffmpeg，请安装后再试")
        print("    运行 install_funasr.bat 安装 ffmpeg")
    
    if not all_modules_ok:
        print("[!] 缺少必要的 Python 模块，请安装后再试")
        print("    运行 install_funasr.bat 安装所有依赖")
    
    if ffmpeg_ok and all_modules_ok:
        print("[✓] 所有依赖都已正确安装，系统已准备就绪")
    
    print("\n按 Enter 键退出...")
    input()

if __name__ == "__main__":
    main()
