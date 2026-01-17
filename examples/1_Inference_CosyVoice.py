import sys
import os
# Add the third_party/Matcha-TTS directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party', 'Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import pygame
import time

def play_audio(file_path):
    """Play audio file using pygame"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        print(f"✅ 播放完成: {file_path}")
    except Exception as e:
        print(f"❌ 播放失败: {e}")
    finally:
        pygame.mixer.quit()

print("🎵 CosyVoice TTS Demo")
print("=" * 50)

# Use the correct local model path and disable JIT loading
print("📥 Loading CosyVoice model...")
cosyvoice = CosyVoice(r'.\models\CosyVoice-300M', load_jit=False, load_onnx=False, fp16=True)

# sft usage
available_speakers = cosyvoice.list_avaliable_spks()
print("🎤 Available speakers:", available_speakers)

# Test different voices with different texts
test_cases = [
    {"text": "你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？", "speaker": "中文女", "filename": "demo_female"},
    {"text": "欢迎使用CosyVoice语音合成系统，我可以用不同的声音为您朗读文本。", "speaker": "中文男", "filename": "demo_male"},
    {"text": "今天天气很好，适合出去散步和享受阳光。", "speaker": "粤语女", "filename": "demo_cantonese"},
]

print("\n🎯 Generating speech samples...")
generated_files = []

for i, case in enumerate(test_cases):
    print(f"\n📝 Processing: {case['text'][:20]}...")
    print(f"🎭 Speaker: {case['speaker']}")
    
    try:
        for j, output in enumerate(cosyvoice.inference_sft(case['text'], case['speaker'], stream=False)):
            filename = f"{case['filename']}_{j}.wav"
            torchaudio.save(filename, output['tts_speech'], 22050)
            generated_files.append(filename)
            print(f"✅ Generated: {filename}")
    except Exception as e:
        print(f"❌ Error generating {case['filename']}: {e}")

print(f"\n🎵 Generated {len(generated_files)} audio files!")

# Play all generated files
print("\n🔊 Playing generated audio files...")
for file in generated_files:
    print(f"\n▶️  Playing: {file}")
    play_audio(file)

print("\n✅ Demo completed successfully!")

# Comment out zero-shot until we have the required audio file
# prompt_speech_16k = load_wav('vocal_3.mp3_10.wav_0006151680_0006360320.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜', '可以动动你的小手点个关注，感谢各位好哥哥，如果之后有新消息，我还会在更新呢。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-25Hz') # or change to pretrained_models/CosyVoice-300M for 50Hz inference
# # zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
# prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], 22050)
# # cross_lingual usage
# prompt_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_cross_lingual('<|en|>And then later on, fully acquiring that company. So keeping management in line, interest in line with the asset that\'s coming into the family is a reason why sometimes we don\'t buy the whole thing.', prompt_speech_16k, stream=False)):
#     torchaudio.save('cross_lingual_{}.wav'.format(i), j['tts_speech'], 22050)
# # vc usage
# prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)
# source_speech_16k = load_wav('cross_lingual_prompt.wav', 16000)
# for i, j in enumerate(cosyvoice.inference_vc(source_speech_16k, prompt_speech_16k, stream=False)):
#     torchaudio.save('vc_{}.wav'.format(i), j['tts_speech'], 22050)

# cosyvoice = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# # instruct usage, support <laughter></laughter><strong></strong>[laughter][breath]
# for i, j in enumerate(cosyvoice.inference_instruct('在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。', '中文男', 'Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.', stream=False)):
#     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], 22050)