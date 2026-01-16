#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 GE Healthcare
Author: jiheng.zhang@gehealthcare.com
SSO: 212597558
"""

import requests
import json

url = "http://192.168.0.112:11434/api/chat"
payload = {
    "model": "qwen3:14b",
    "messages": [
        {"role": "user", "content": "Hello, how are you? introduce yourself"}
    ]
}

def process_stream_response(response):
    accumulated_content = ""
    
    try:
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if 'message' in data:
                    message_content = data['message'].get('content', '')
                    accumulated_content += message_content
                    
                if data.get('done', False):
                    break
                    
        return accumulated_content.strip()
        
    except Exception as e:
        print(f"Error processing stream: {str(e)}")
        return None

try:
    response = requests.post(url, json=payload, stream=True)
    response.raise_for_status()
    
    final_content = process_stream_response(response)
    if final_content:
        print("\nResponse:", final_content)
    else:
        print("No valid response received")
        
except requests.exceptions.RequestException as e:
    print(f"Request failed: {str(e)}")