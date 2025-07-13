import os
import time
import wave
import threading
import queue
import numpy as np
import pyaudio
from datetime import datetime
import colorama
from colorama import Fore, Back, Style

# 初始化 colorama 以支持颜色输出
colorama.init()

try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
    print(f"{Fore.GREEN}[成功]{Style.RESET_ALL} FunASR 模块已加载")
except ImportError:
    FUNASR_AVAILABLE = False
    print(f"{Fore.RED}[警告]{Style.RESET_ALL} FunASR 模块未找到，将使用基本的振幅检测")

class KeywordActivatedRecorder:
    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1, 
                 format=pyaudio.paInt16, threshold=0.03, 
                 silence_duration=2.0, min_speech_duration=1.0,
                 buffer_duration=5.0):
        """
        初始化录音器
        
        参数:
            sample_rate: 采样率 (Hz)
            chunk_size: 每次处理的音频块大小
            channels: 通道数
            format: 音频格式
            threshold: 音频激活阈值（用于备用检测）
            silence_duration: 停止录音前的静音持续时间 (秒)
            min_speech_duration: 最小语音持续时间以被认为是有效语音 (秒)
            buffer_duration: 预缓冲持续时间 (秒)，用于捕获关键词之前的音频
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format = format
        self.threshold = threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        
        # 计算预缓冲区大小
        self.buffer_frames = int(buffer_duration * sample_rate / chunk_size)
        self.buffer = queue.Queue(maxsize=self.buffer_frames)
        
        # 录音状态
        self.recording = False
        self.speech_detected = False
        self.silence_counter = 0
        self.speech_counter = 0
        self.max_silence_frames = int(silence_duration * sample_rate / chunk_size)
        self.min_speech_frames = int(min_speech_duration * sample_rate / chunk_size)
        
        # 音频对象
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        
        # 检查音频设备
        self.check_audio_devices()
        
        # 线程锁
        self.lock = threading.Lock()
        
        # FunASR 模型初始化
        self.vad_model = None
        self.asr_model = None
        self.use_funasr = FUNASR_AVAILABLE
        
        if self.use_funasr:
            self.init_funasr_models()
        else:
            print(f"{Fore.YELLOW}[信息]{Style.RESET_ALL} 使用基本振幅检测算法")
        
        # 关键词检测相关变量
        self.keyword_buffer = queue.Queue(maxsize=int(4.0 * sample_rate / chunk_size))  # 4秒关键词缓冲
        self.keyword_detected = False
        self.speech_segments = []  # 存储检测到的语音段
        self.keyword_candidates = ["你好", "小助手", "开始录音", "录音开始", "小爱", "小度"]
        self.current_keyword = "你好"  # 默认关键词
        
        # VAD 相关变量
        self.vad_cache = {}
        self.accumulated_audio = np.array([], dtype=np.float32)
        self.last_vad_check = time.time()
        self.vad_check_interval = 3.0  # 每3秒进行一次VAD检查
        # 使用16000采样点（1秒）的窗口，确保是400的倍数
        self.vad_window_size = 16000  # 1秒窗口
        self.audio_buffer_for_vad = np.array([], dtype=np.float32)
        
        print(f"{Fore.YELLOW}[关键词]{Style.RESET_ALL} 当前激活关键词: '{self.current_keyword}'")
        print(f"{Fore.CYAN}[提示]{Style.RESET_ALL} 使用 FunASR VAD+ASR 进行关键词检测")
    
    def init_funasr_models(self):
        """初始化 FunASR VAD 和 ASR 模型"""
        try:
            print(f"{Fore.CYAN}[初始化]{Style.RESET_ALL} 正在加载 FunASR VAD 模型...")
            # 加载 VAD 模型
            self.vad_model = AutoModel(
                model="fsmn-vad",
                model_revision="v2.0.4"
            )
            print(f"{Fore.GREEN}[成功]{Style.RESET_ALL} FunASR VAD 模型加载完成")
            
            print(f"{Fore.CYAN}[初始化]{Style.RESET_ALL} 正在加载 FunASR ASR 模型...")
            # 加载 ASR 模型用于关键词识别
            self.asr_model = AutoModel(
                model="paraformer-zh",
                model_revision="v2.0.4"
            )
            print(f"{Fore.GREEN}[成功]{Style.RESET_ALL} FunASR ASR 模型加载完成")
            
        except Exception as e:
            print(f"{Fore.RED}[错误]{Style.RESET_ALL} FunASR 模型加载失败: {e}")
            self.use_funasr = False
            self.vad_model = None
            self.asr_model = None

    def start(self):
        """开始监听音频"""
        print(f"{Fore.CYAN}[系统]{Style.RESET_ALL} 启动音频监听...")
        self.stream = self.pyaudio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        print(f"{Fore.GREEN}[就绪]{Style.RESET_ALL} 语音激活系统已启动，正在等待语音...")
        
    def stop(self):
        """停止监听并关闭资源"""
        print(f"{Fore.CYAN}[系统]{Style.RESET_ALL} 正在关闭音频系统...")
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio.terminate()
        print(f"{Fore.GREEN}[完成]{Style.RESET_ALL} 音频系统已关闭")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数，处理输入的音频数据"""
        # 将二进制数据转换为数组
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # 计算音频振幅 (归一化)
        amplitude = np.abs(audio_data).mean() / 32767.0
        
        # 添加到预缓冲区
        try:
            if self.buffer.full():
                self.buffer.get_nowait()
            self.buffer.put_nowait(in_data)
        except queue.Full:
            pass
        
        # VAD 检测
        is_speech = amplitude > self.threshold  # 默认使用振幅
        
        # 如果 FunASR VAD 可用，使用它进行检测
        if self.vad_model is not None:
            try:
                # 将二进制数据转换为 float32 并归一化
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # 添加到VAD专用缓冲区
                self.audio_buffer_for_vad = np.concatenate([self.audio_buffer_for_vad, audio_float])
                
                # 定期进行 VAD 检测
                current_time = time.time()
                if current_time - self.last_vad_check >= self.vad_check_interval and not self.recording:
                    self.last_vad_check = current_time
                    
                    # 检查是否有足够的音频数据进行VAD
                    if len(self.audio_buffer_for_vad) >= self.vad_window_size:
                        # 确保窗口大小是400的精确倍数
                        window_samples = (self.vad_window_size // 400) * 400
                        vad_window = self.audio_buffer_for_vad[:window_samples]
                        
                        try:
                            # 使用 VAD 模型进行语音检测
                            vad_result = self.vad_model.generate(
                                input=vad_window.reshape(1, -1),
                                chunk_size=400  # 指定分块大小
                            )
                            
                            # 检查 VAD 结果
                            if isinstance(vad_result, list) and len(vad_result) > 0:
                                result_dict = vad_result[0]
                                if isinstance(result_dict, dict) and 'value' in result_dict:
                                    vad_segments = result_dict['value']
                                    
                                    # 如果检测到语音段，处理它们
                                    if vad_segments and not self.recording and not self.keyword_detected:
                                        print(f"\n{Fore.CYAN}[VAD]{Style.RESET_ALL} 检测到 {len(vad_segments)} 个语音段")
                                        
                                        # 针对每个检测到的语音段提取音频
                                        for segment in vad_segments:
                                            if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                                                start_time, end_time = segment[0], segment[1]
                                                # 时间单位转换为采样点
                                                start_sample = int(start_time * self.sample_rate)
                                                end_sample = int(end_time * self.sample_rate)
                                                
                                                # 检查索引范围
                                                if start_sample < 0:
                                                    start_sample = 0
                                                if end_sample > len(vad_window):
                                                    end_sample = len(vad_window)
                                                
                                                # 提取音频段并进行关键词识别
                                                if end_sample > start_sample:
                                                    segment_audio = vad_window[start_sample:end_sample]
                                                    if len(segment_audio) >= 1600:  # 至少100ms
                                                        # 使用线程进行异步关键词识别
                                                        threading.Thread(
                                                            target=self.recognize_keyword,
                                                            args=(segment_audio,),
                                                            daemon=True
                                                        ).start()
                                                        
                                                        # 检查振幅，如果足够高也认为是语音
                                                        segment_amplitude = np.abs(segment_audio).mean()
                                                        if segment_amplitude > self.threshold * 1.5:
                                                            is_speech = True
                                            else:
                                                print(f"\n{Fore.YELLOW}[VAD]{Style.RESET_ALL} 无效语音段格式: {segment}")
                        except Exception as vad_error:
                            print(f"\n{Fore.RED}[VAD处理错误]{Style.RESET_ALL} {vad_error}")
                            # 如果VAD处理失败，退回到基于振幅的检测
                            is_speech = amplitude > self.threshold
                            
                            # 如果振幅足够大，也尝试直接用ASR
                            if amplitude > 0.03 and not self.recording and not self.keyword_detected:
                                # 用最近的一段音频直接尝试ASR关键词识别
                                recent_audio = self.audio_buffer_for_vad[-16000:] if len(self.audio_buffer_for_vad) > 16000 else self.audio_buffer_for_vad
                                if len(recent_audio) >= 8000:  # 至少0.5秒
                                    threading.Thread(
                                        target=self.recognize_keyword,
                                        args=(recent_audio,),
                                        daemon=True
                                    ).start()
                        
                        # 保留一部分音频作为下次检测的上下文
                        overlap_size = window_samples // 3  # 保留33%的重叠
                        self.audio_buffer_for_vad = self.audio_buffer_for_vad[-overlap_size:]
                    
                    # 限制缓冲区大小，防止内存过大
                    max_buffer_size = self.vad_window_size * 2  # 最多保留2个窗口的数据
                    if len(self.audio_buffer_for_vad) > max_buffer_size:
                        self.audio_buffer_for_vad = self.audio_buffer_for_vad[-max_buffer_size:]
                        
            except Exception as e:
                print(f"\n{Fore.RED}[VAD错误]{Style.RESET_ALL} {e}")
                # 回退到振幅检测
                is_speech = amplitude > self.threshold
        else:
            # 如果 FunASR 不可用，使用振幅检测
            is_speech = amplitude > self.threshold
        
        # 显示音频电平和状态
        vad_status = "🎤 语音" if is_speech else "🔇 静音"
        vad_color = Fore.GREEN if is_speech else Fore.BLUE
        
        if amplitude > 0.001:  # 只在有音频输入时显示
            level_bars = int(amplitude * 30)
            level_display = '█' * level_bars + '░' * (30 - level_bars)
            status_line = f"{vad_color}{vad_status}{Style.RESET_ALL} | 电平: [{level_display}] {amplitude:.3f}"
            
            # 显示录音状态
            if self.recording:
                status_line += f" | {Fore.RED}● 录音中{Style.RESET_ALL}"
            elif self.keyword_detected:
                status_line += f" | {Fore.YELLOW}● 关键词激活: {self.current_keyword}{Style.RESET_ALL}"
            
            print(f"\r{status_line}", end='', flush=True)
        
        # 状态逻辑处理
        with self.lock:
            if is_speech:
                self.speech_counter += 1
                self.silence_counter = 0
                
                # 关键词激活后开始录音 - 强化激活逻辑
                if self.keyword_detected and not self.recording:
                    if self.speech_counter >= self.min_speech_frames:
                        print(f"\n{Fore.GREEN}[录音]{Style.RESET_ALL} 关键词 '{self.current_keyword}' 激活 - 开始录音")
                        self.start_recording()
                    else:
                        print(f"\r{Fore.YELLOW}[等待]{Style.RESET_ALL} 等待足够的语音持续时间: {self.speech_counter}/{self.min_speech_frames}", end='', flush=True)
            else:
                self.silence_counter += 1
                
                # 如果静音足够长并且正在录音，则停止录音
                if self.silence_counter >= self.max_silence_frames and self.recording:
                    self.stop_recording()
                    
                # 如果语音不够长，重置语音计数器
                if not self.recording and self.speech_counter < self.min_speech_frames:
                    self.speech_counter = 0
        
        return (in_data, pyaudio.paContinue)
    
    def process_vad_segments(self, vad_segments, audio_data):
        """处理 VAD 检测到的语音段（已简化，主要由recognize_keyword处理）"""
        # 这个方法现在主要用于调试信息
        for segment in vad_segments:
            try:
                if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                    start_time, end_time = segment[0], segment[1]
                    print(f"\n{Fore.CYAN}[VAD段]{Style.RESET_ALL} 时间: {start_time:.2f}s - {end_time:.2f}s")
            except Exception as e:
                print(f"\n{Fore.YELLOW}[VAD段警告]{Style.RESET_ALL} 处理段信息时出错: {e}")

    def recognize_keyword(self, audio_segment):
        """使用 ASR 识别语音段中的关键词"""
        if self.asr_model is None or self.keyword_detected or self.recording:
            return
            
        try:
            # 确保音频段格式正确
            if len(audio_segment.shape) == 1:
                audio_input = audio_segment.reshape(1, -1)
            else:
                audio_input = audio_segment
            
            # 使用 ASR 模型进行语音识别
            asr_result = self.asr_model.generate(
                input=audio_input,
                cache={},
                language="zh"
            )
            
            # 解析识别结果
            if isinstance(asr_result, list) and len(asr_result) > 0:
                result_dict = asr_result[0]
                if isinstance(result_dict, dict):
                    recognized_text = result_dict.get('text', '').strip()
                    
                    if recognized_text:
                        print(f"\n{Fore.BLUE}[识别]{Style.RESET_ALL} 语音识别结果: '{recognized_text}'")
                        
                        # 检查是否包含关键词
                        for keyword in self.keyword_candidates:
                            if keyword in recognized_text:
                                with self.lock:
                                    if not self.keyword_detected and not self.recording:
                                        print(f"\n{Fore.GREEN}[关键词]{Style.RESET_ALL} 检测到关键词 '{keyword}' - 已激活")
                                        self.keyword_detected = True
                                        self.current_keyword = keyword
                                        self.speech_counter = max(self.speech_counter, self.min_speech_frames // 2)  # 加快激活速度
                                        return
                    else:
                        print(f"\n{Fore.YELLOW}[ASR]{Style.RESET_ALL} 识别结果为空")
            
        except Exception as e:
            print(f"\n{Fore.RED}[ASR错误]{Style.RESET_ALL} 语音识别失败: {e}")
    
    def start_recording(self):
        """开始录音"""
        with self.lock:
            if not self.recording:
                self.recording = True
                self.frames = []
                
                # 添加预缓冲区中的所有帧
                while not self.buffer.empty():
                    self.frames.append(self.buffer.get())
                
                print(f"\n{Fore.GREEN}[录音]{Style.RESET_ALL} 关键词激活成功 - 开始录音...")
                print(f"{Fore.CYAN}[状态]{Style.RESET_ALL} 请继续说话，静音 {self.silence_duration} 秒后自动停止")
                
                # 重置计数器和状态
                self.silence_counter = 0
                # 清空关键词缓冲区，为下次检测做准备
                while not self.keyword_buffer.empty():
                    try:
                        self.keyword_buffer.get_nowait()
                    except queue.Empty:
                        break
    
    def stop_recording(self):
        """停止录音并保存文件"""
        with self.lock:
            if self.recording:
                self.recording = False
                print(f"\n{Fore.YELLOW}[录音]{Style.RESET_ALL} 检测到静音 - 停止录音")
                
                # 如果有足够的帧，保存录音
                if len(self.frames) > 0:
                    threading.Thread(target=self.save_recording).start()
                
                # 重置计数器和状态
                self.speech_counter = 0
                self.speech_detected = False
                self.keyword_detected = False  # 重置关键词状态，准备下次检测
                
                print(f"{Fore.CYAN}[等待]{Style.RESET_ALL} 等待下一个关键词 '{self.current_keyword}'...")
    
    def save_recording(self):
        """将录音保存为 WAV 文件"""
        if not self.frames:
            print(f"{Fore.RED}[错误]{Style.RESET_ALL} 没有录音数据可保存")
            return
            
        # 确保存在录音目录
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
            
        # 使用时间戳创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("recordings", f"recording_{timestamp}.wav")
        
        print(f"{Fore.BLUE}[保存]{Style.RESET_ALL} 正在保存录音到: {filename}")
        
        # 保存为 WAV 文件
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
             
        duration = len(self.frames) * self.chunk_size / self.sample_rate
        file_size = os.path.getsize(filename) / 1024  # KB
        
        print(f"{Fore.GREEN}[完成]{Style.RESET_ALL} 录音已保存! 文件: {filename}")
        print(f"{Fore.CYAN}[信息]{Style.RESET_ALL} 录音时长: {duration:.2f} 秒, 文件大小: {file_size:.2f} KB")
        
        # 清空帧
        self.frames = []

    def process_chunk(self, in_data):
        """处理单个音频块（用于非回调模式）"""
        if self.recording:
            self.frames.append(in_data)
    
    def check_audio_devices(self):
        """检查可用的音频设备"""
        device_count = self.pyaudio.get_device_count()
        print(f"{Fore.BLUE}[设备]{Style.RESET_ALL} 检测到 {device_count} 个音频设备:")
        
        input_devices = []
        for i in range(device_count):
            try:
                info = self.pyaudio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_devices.append((i, info['name']))
                    print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {info['name']} (输入通道: {info['maxInputChannels']})")
            except Exception as e:
                continue
        
        if not input_devices:
            raise Exception("没有找到可用的音频输入设备")
        
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            print(f"{Fore.GREEN}[默认]{Style.RESET_ALL} 使用默认输入设备: {default_input['name']}")
        except Exception as e:
            print(f"{Fore.YELLOW}[警告]{Style.RESET_ALL} 无法获取默认输入设备: {e}")
        
        print(f"{Fore.CYAN}[就绪]{Style.RESET_ALL} 找到 {len(input_devices)} 个可用输入设备\n")
    
    def detect_keyword_in_audio(self, audio_data):
        """
        备用的关键词检测算法（当 FunASR 不可用时使用）
        """
        try:
            # 计算音频特征
            amplitude = np.abs(audio_data).mean()
            audio_length = len(audio_data) / self.sample_rate
            
            # 简化的启发式检测
            if 0.4 <= audio_length <= 2.0 and amplitude > self.threshold * 2:
                mid_point = len(audio_data) // 2
                first_half_energy = np.sum(audio_data[:mid_point].astype(float) ** 2)
                second_half_energy = np.sum(audio_data[mid_point:].astype(float) ** 2)
                
                energy_ratio = min(first_half_energy, second_half_energy) / max(first_half_energy, second_half_energy)
                
                if energy_ratio > 0.3:
                    return True, f"检测到可能的关键词 (时长:{audio_length:.2f}s, 能量比:{energy_ratio:.2f})"
            
            return False, f"音频不匹配关键词模式 (时长:{audio_length:.2f}s, 振幅:{amplitude:.3f})"
            
        except Exception as e:
            return False, f"关键词检测错误: {e}"

    def process_keyword_detection(self, audio_chunk):
        """处理关键词检测（备用方法，当 FunASR 不可用时使用）"""
        if self.use_funasr:
            return False  # 使用 FunASR 时不使用此方法
            
        # 将音频块添加到关键词缓冲区
        try:
            if self.keyword_buffer.full():
                self.keyword_buffer.get_nowait()
            
            # 转换为 numpy 数组
            if isinstance(audio_chunk, bytes):
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            else:
                audio_array = audio_chunk
                
            self.keyword_buffer.put_nowait(audio_array)
        except queue.Full:
            pass
        
        # 收集缓冲区中的音频用于关键词检测
        if not self.keyword_buffer.empty():
            buffer_data = []
            temp_queue = queue.Queue()
            
            while not self.keyword_buffer.empty():
                try:
                    chunk = self.keyword_buffer.get_nowait()
                    buffer_data.append(chunk)
                    temp_queue.put(chunk)
                except queue.Empty:
                    break
            
            # 恢复队列状态
            while not temp_queue.empty():
                try:
                    self.keyword_buffer.put_nowait(temp_queue.get_nowait())
                except queue.Full:
                    break
            
            # 合并音频数据进行关键词检测
            if len(buffer_data) > 5:
                combined_audio = np.concatenate(buffer_data)
                is_keyword, message = self.detect_keyword_in_audio(combined_audio)
                
                if is_keyword and not self.keyword_detected and not self.recording:
                    print(f"\n{Fore.GREEN}[关键词]{Style.RESET_ALL} {message}")
                    print(f"{Fore.GREEN}[激活]{Style.RESET_ALL} 检测到关键词 '{self.current_keyword}' - 准备开始录音")
                    self.keyword_detected = True
                    return True
        
        return False

if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}           FunASR VAD+ASR 关键词激活录音系统{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}工作模式: FunASR VAD+ASR 关键词激活录音{Style.RESET_ALL}")
    print(f"  • 默认关键词: {Fore.GREEN}'你好'{Style.RESET_ALL}")
    print(f"  • 支持关键词: 你好, 小助手, 开始录音, 录音开始, 小爱, 小度")
    print(f"  • 检测方式: VAD检测语音段 + ASR识别关键词")
    print(f"{Fore.YELLOW}系统配置:{Style.RESET_ALL}")
    print(f"  • VAD 检查间隔: 3.0 秒")
    print(f"  • VAD 窗口大小: 1.0秒 (16000 采样点)")
    print(f"  • 静音超时: 2.0 秒")
    print(f"  • 最小语音时长: 0.8 秒") 
    print(f"  • 预缓冲时长: 5.0 秒")
    print(f"{Fore.RED}重要: 系统使用固定400分块大小的VAD检测！{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}控制说明: 按 Ctrl+C 退出程序{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}{Style.RESET_ALL}\n")
    
    # 创建并启动录音器
    try:
        recorder = KeywordActivatedRecorder(
            threshold=0.03,              # 备用振幅阈值
            silence_duration=2.0,        # 停止录音前的静音持续时间 (秒)
            min_speech_duration=0.5,     # 最小语音持续时间以便开始录音 (秒)
            buffer_duration=5.0          # 预缓冲持续时间 (秒)
        )
        
        recorder.start()
        print(f"{Fore.GREEN}[状态]{Style.RESET_ALL} FunASR VAD+ASR 关键词检测系统已启动")
        print(f"{Fore.CYAN}[等待]{Style.RESET_ALL} 请说出关键词 '{Fore.GREEN}你好{Style.RESET_ALL}' 来激活录音")
        print(f"{Fore.BLUE}[提示]{Style.RESET_ALL} 系统会实时显示VAD检测状态和语音识别结果\n")
        
        # 保持程序运行
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[系统]{Style.RESET_ALL} 检测到键盘中断，正在关闭...")
    except Exception as e:
        print(f"\n{Fore.RED}[错误]{Style.RESET_ALL} 系统错误: {e}")
    finally:
        try:
            recorder.stop()
        except:
            pass
        print(f"{Fore.GREEN}[完成]{Style.RESET_ALL} 程序已安全退出")
            
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[系统]{Style.RESET_ALL} 检测到键盘中断，正在关闭...")
    except Exception as e:
        print(f"\n{Fore.RED}[错误]{Style.RESET_ALL} 系统错误: {e}")
    finally:
        try:
            recorder.stop()
        except:
            pass
        print(f"{Fore.GREEN}[完成]{Style.RESET_ALL} 程序已安全退出")
