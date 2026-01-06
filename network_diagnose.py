#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import subprocess
import json
import os
import sys
import platform
from urllib.parse import urlparse

def run_command(cmd):
    """运行系统命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def check_routing_table():
    """检查路由表"""
    print("=" * 50)
    print("路由表检查")
    print("=" * 50)
    
    if platform.system() == "Windows":
        stdout, stderr, code = run_command("route print")
        if code == 0:
            print(stdout)
        else:
            print(f"错误: {stderr}")
    else:
        stdout, stderr, code = run_command("netstat -rn")
        if code == 0:
            print(stdout)
        else:
            print(f"错误: {stderr}")
    
    print("\n")

def check_network_interfaces():
    """检查网络接口"""
    print("=" * 50)
    print("网络接口检查")
    print("=" * 50)
    
    if platform.system() == "Windows":
        stdout, stderr, code = run_command("ipconfig /all")
        if code == 0:
            print(stdout)
        else:
            print(f"错误: {stderr}")
    else:
        stdout, stderr, code = run_command("ifconfig -a")
        if code == 0:
            print(stdout)
        else:
            print(f"错误: {stderr}")
    
    print("\n")

def check_proxy_settings():
    """检查代理设置"""
    print("=" * 50)
    print("代理设置检查")
    print("=" * 50)
    
    # 检查环境变量中的代理设置
    proxy_vars = [
        'http_proxy', 'https_proxy', 'ftp_proxy', 
        'HTTP_PROXY', 'HTTPS_PROXY', 'FTP_PROXY', 'ALL_PROXY'
    ]
    
    proxy_configured = False
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"{var}: {value}")
            proxy_configured = True
    
    if not proxy_configured:
        print("未在环境变量中找到代理设置")
    
    # Windows系统检查注册表中的代理设置
    if platform.system() == "Windows":
        print("\n检查Windows代理设置:")
        stdout, stderr, code = run_command(
            'reg query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings" /v ProxyEnable'
        )
        if code == 0:
            if "0x1" in stdout:
                print("代理已启用")
            else:
                print("代理未启用")
        else:
            print(f"无法检查注册表项: {stderr}")
        
        stdout, stderr, code = run_command(
            'reg query "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings" /v ProxyServer'
        )
        if code == 0:
            print(f"代理服务器设置: {stdout.split()[-1] if stdout.strip() else 'None'}")
        else:
            print(f"无法检查代理服务器设置: {stderr}")
    
    print("\n")

def check_dns_settings():
    """检查DNS设置"""
    print("=" * 50)
    print("DNS设置检查")
    print("=" * 50)
    
    if platform.system() == "Windows":
        stdout, stderr, code = run_command("nslookup google.com")
        if code == 0:
            print(stdout)
        else:
            print(f"DNS查询失败: {stderr}")
    else:
        stdout, stderr, code = run_command("dig google.com")
        if code == 0:
            print(stdout)
        else:
            print(f"DNS查询失败: {stderr}")
    
    print("\n")

def check_network_connectivity():
    """检查网络连通性"""
    print("=" * 50)
    print("网络连通性检查")
    print("=" * 50)
    
    test_hosts = ["8.8.8.8", "google.com", "baidu.com"]
    
    for host in test_hosts:
        if platform.system() == "Windows":
            stdout, stderr, code = run_command(f"ping -n 1 {host}")
        else:
            stdout, stderr, code = run_command(f"ping -c 1 {host}")
        
        if code == 0:
            print(f"✓ 可以访问 {host}")
        else:
            print(f"✗ 无法访问 {host}")
    
    print("\n")

def main():
    """主函数"""
    print("网络诊断工具")
    print("系统: " + platform.system() + " " + platform.release())
    print("Python版本: " + sys.version)
    print("\n")
    
    check_routing_table()
    check_network_interfaces()
    check_proxy_settings()
    check_dns_settings()
    check_network_connectivity()
    
    print("=" * 50)
    print("诊断完成")
    print("=" * 50)

if __name__ == "__main__":
    main()