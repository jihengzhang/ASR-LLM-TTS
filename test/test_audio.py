import pygame
import time

def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)  # 等待音频播放结束
        print(f"播放完成: {file_path}")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

print('Testing audio playback...')
play_audio('prompt_sft_0.wav')
play_audio('sft_0.wav')
print('Audio test complete!')
