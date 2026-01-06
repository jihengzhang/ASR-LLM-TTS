#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import queue
import threading
import time

# 创建队列并添加数据
q = queue.Queue()
for i in range(5):
    q.put(f"数据{i}")

def worker(thread_id):
    while True:
        try:
            # 从队列获取数据
            data = q.get(timeout=1)  # 设置超时避免无限等待
            print(f"线程{thread_id} 获取到: {data}")
            wait_time = 0.5
            q.task_done()
        except queue.Empty:
            break

# 创建多个线程
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()
    # worker(i)


# 等待所有线程完成
for t in threads:
    t.join()