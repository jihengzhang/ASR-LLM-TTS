import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import argparse
import logging
from typing import Tuple, Optional
from collections import namedtuple

# Define a named tuple for audio endpoints
AudioEndpoint = namedtuple('AudioEndpoint', ['idx', 'name', 'mode'])

class AudioDeviceManager:
    def __init__(self, device_name: str = "ESP-ADF-AUD"):
        # 常量定义
        self.HFP_KEYWORDS = ['hands-free', 'headset', 'communication', 'bthhfenum.sys']
        self.A2DP_KEYWORDS = ['stereo', 'btha2dp.sys']
        self.HOSTAPI_NAMES = {
            0: 'MME',
            1: 'DirectSound',
            2: 'WASAPI',
            3: 'ASIO'
        }
        
        self.device_name = device_name
        self.devices = sd.query_devices()
        self.input_idx = None
        self.output_idx = None
        
        # Configuration
        # self.sample_rates = [44100, 16000, 8000]
        self.channels = 1
        self.chunk_size = 1024
        
        # HFP settings
        self.hfp_sample_rate = 8000 #48000 # 44100 #16000
        self.hfp_keepalive_interval = 0.5
        self.hfp_activation_delay = 1.0
        
        # Initialize device endpoints
        self._find_device_endpoints()
    
    # def _is_target_device(self, name: str) -> bool:
    #     """检查设备名称是否匹配目标设备"""
    #     return self.device_name.lower() in name.lower()
    
    def _is_target_device(self, dev: dict) -> bool:
        """检查设备是否匹配目标设备（名称匹配且不是ASIO）"""
        name_match = self.device_name.lower() in dev['name'].lower()
        not_asio = dev['hostapi'] != 3  # ASIO = 3
        return name_match and not_asio
    
    def _is_hfp_device(self, name: str) -> bool:
        """检查是否为HFP设备"""
        return any(kw in name.lower() for kw in self.HFP_KEYWORDS)
    
    def _is_a2dp_device(self, name: str) -> bool:
        """检查是否为A2DP设备"""
        return any(kw in name.lower() for kw in self.A2DP_KEYWORDS)
    
    def _get_device_mode(self, dev_info: dict) -> str:
        """确定设备模式（HFP或A2DP）"""
        name = dev_info['name'].lower()
        if any(kw in name for kw in self.HFP_KEYWORDS):
            return 'HFP'
        elif any(kw in name for kw in self.A2DP_KEYWORDS):
            return 'A2DP'
        return 'UNKNOWN'
    
    def _is_rate_supported(self, device_info: dict, rate: int) -> bool:
        """检查采样率是否被设备支持"""
        if device_info.get('default_samplerate') == rate:
            return True
        supported_rates = device_info.get('supported_samplerates')
        return supported_rates is None or rate in supported_rates
    
    def _format_device_info(self, idx: int, dev: dict, is_target: bool = False) -> str:
        """格式化设备信息显示"""
        dev_type = []
        if dev['max_input_channels'] > 0:
            dev_type.append('IN')
        if dev['max_output_channels'] > 0:
            dev_type.append('OUT')
        
        sr = int(dev['default_samplerate']) if dev['default_samplerate'] else 'N/A'
        marker = '*' if is_target else ' '
        api_name = self.HOSTAPI_NAMES.get(dev['hostapi'], f'API{dev["hostapi"]}')
        return f"{marker}{idx:2d}  {api_name:<11} {'+'.join(dev_type):<6} {sr:8} {dev['name']}"
    
    def _find_device_endpoints(self):
        """查找并设置设备端点"""
        endpoints = {
            'hfp': {'input': None, 'output': None},
            'a2dp': {'input': None, 'output': None},
            'fallback': {'input': None, 'output': None}
        }
        
        # 初始化变量，避免UnboundLocalError
        input_endpoint = None
        output_endpoint = None
        
        # 按API类型分组目标设备
        target_devices_by_api = {}
        
        for idx, dev in enumerate(self.devices):
            name = dev['name']
            if not self._is_target_device(dev):
                continue
                
            api_id = dev['hostapi']
            if api_id not in target_devices_by_api:
                target_devices_by_api[api_id] = []
            target_devices_by_api[api_id].append((idx, dev))
        
        # 优先选择WASAPI，然后DirectSound，最后MME
        preferred_apis = [2, 1, 0]  # WASAPI, DirectSound, MME
        
        for api_id in preferred_apis:
            if api_id not in target_devices_by_api:
                continue
                
            devices_in_api = target_devices_by_api[api_id]
            
            # 处理当前API下的所有设备
            for idx, dev in devices_in_api:
                name = dev['name']
                
                for mode, condition in [
                    ('hfp', self._is_hfp_device(name)),
                    ('a2dp', self._is_a2dp_device(name)),
                    ('fallback', True)
                ]:
                    if condition:
                        if dev['max_input_channels'] > 0 and endpoints[mode]['input'] is None:
                            endpoints[mode]['input'] = AudioEndpoint(idx, name, mode.upper())
                        if dev['max_output_channels'] > 0 and endpoints[mode]['output'] is None:
                            endpoints[mode]['output'] = AudioEndpoint(idx, name, mode.upper())
                        break
        
            # 在处理完当前API的所有设备后，检查是否找到合适的配对
            input_endpoint = (endpoints['hfp']['input'] or 
                             endpoints['a2dp']['input'] or 
                             endpoints['fallback']['input'])
            output_endpoint = (endpoints['hfp']['output'] or 
                              endpoints['a2dp']['output'] or 
                              endpoints['fallback']['output'])
            
            if input_endpoint and output_endpoint:
                input_api = self.devices[input_endpoint.idx]['hostapi']
                output_api = self.devices[output_endpoint.idx]['hostapi']
                
                if input_api == output_api:
                    break  # 找到同一API下的设备对，停止搜索其他API
                elif input_endpoint or output_endpoint:
                    # 至少找到一个设备也可以
                    break
    
        self.input_idx = input_endpoint.idx if input_endpoint else None
        self.output_idx = output_endpoint.idx if output_endpoint else None
    
        # Print found endpoints
        if input_endpoint or output_endpoint:
            print("\nFound device endpoints:")
            if input_endpoint:
                input_api_name = self.HOSTAPI_NAMES.get(self.devices[input_endpoint.idx]['hostapi'], 'Unknown')
                print(f"Input [{input_endpoint.idx}] {input_endpoint.name} ({input_endpoint.mode}) - {input_api_name}")
            if output_endpoint:
                output_api_name = self.HOSTAPI_NAMES.get(self.devices[output_endpoint.idx]['hostapi'], 'Unknown')
                print(f"Output [{output_endpoint.idx}] {output_endpoint.name} ({output_endpoint.mode}) - {output_api_name}")

    def _activate_hfp_mode(self, device_idx: int) -> bool:
        """激活HFP模式并建立稳定连接"""
        try:
            with sd.OutputStream(device=device_idx,
                               samplerate=self.hfp_sample_rate,
                               channels=self.channels,
                               dtype=np.int16,
                               blocksize=self.chunk_size) as stream:
                zeros = np.zeros((self.chunk_size, self.channels), dtype=np.int16)
                stream.start()
                
                # Send initial silence burst
                for _ in range(3):
                    stream.write(zeros)
                    time.sleep(0.1)
                
                time.sleep(self.hfp_activation_delay)
                stream.write(zeros)
                return True
        except Exception as e:
            logging.error(f"HFP activation failed: {e}")
            return False

    def _create_audio_stream(self, input_idx: Optional[int] = None,
                       output_idx: Optional[int] = None,
                       sample_rate: int = 16000,
                       force_hfp: bool = False):
        """创建音频流，支持全双工/半双工模式"""
        if input_idx is None and output_idx is None:
            raise ValueError("At least one input or output device must be specified")
        
        # 检查设备模式并打印信息
        input_mode = None
        output_mode = None
        is_hfp_stream = False
        
        if input_idx is not None:
            input_mode = self._get_device_mode(self.devices[input_idx])
            input_device_name = self.devices[input_idx]['name']
            print(f"Input device [{input_idx}]: {input_device_name}")
            print(f"Input device mode: {input_mode}")
            
        if output_idx is not None:
            output_mode = self._get_device_mode(self.devices[output_idx])
            output_device_name = self.devices[output_idx]['name']
            print(f"Output device [{output_idx}]: {output_device_name}")
            print(f"Output device mode: {output_mode}")
        
        # 判断是否为HFP流
        if force_hfp or input_mode == 'HFP' or output_mode == 'HFP':
            is_hfp_stream = True
            sample_rate = self.hfp_sample_rate
            print(f"🎯 Stream mode: HFP (采样率强制设为 {sample_rate}Hz)")
        else:
            print(f"🎯 Stream mode: Non-HFP (采样率: {sample_rate}Hz)")
        
        # 简化设备配置，避免复杂的组合检查
        if input_idx is not None and output_idx is not None:
            # 全双工模式 - 先尝试使用相同的设备索引
            if input_idx == output_idx:
                device = input_idx  # 同一个设备
                print(f"📱 Device configuration: Same device (ID: {input_idx})")
            else:
                # 不同设备，使用元组格式
                device = (input_idx, output_idx)
                print(f"📱 Device configuration: Different devices (Input: {input_idx}, Output: {output_idx})")
            stream_class = sd.Stream
            print(f"🔄 Stream type: Full-duplex (sd.Stream)")
        elif input_idx is not None:
            # 仅输入
            device = input_idx
            stream_class = sd.InputStream
            print(f"📱 Device configuration: Input only (ID: {input_idx})")
            print(f"🎤 Stream type: Input-only (sd.InputStream)")
        else:
            # 仅输出
            device = output_idx
            stream_class = sd.OutputStream
            print(f"📱 Device configuration: Output only (ID: {output_idx})")
            print(f"🔊 Stream type: Output-only (sd.OutputStream)")
        
        # 创建流参数
        # kwargs = {
        #     # 'device': device,
        #     'device': (3, 8),
        #     # 'samplerate': sample_rate, #use default
        #     'samplerate': 44100, #use default
        #     'channels': self.channels,
        #     'dtype': np.int16,
        #     'blocksize': self.chunk_size
        # }

        kwargs = {
            # 'device': device,
            'device': (2, 4), #（IN OUT) ESP-ADF-AUDIO
            # 'device': (3, 8), #（IN OUT)
            # 'device': (2, 7), # Jabra 810
            # 'samplerate': sample_rate, #use default
            'samplerate': 44100, #use default
            'channels': self.channels,
            'dtype': np.int16,
            'blocksize': self.chunk_size
        }        
        print(f"⚙️ Stream parameters: {kwargs}")
        
        try:
            stream = stream_class(**kwargs)
            logging.info(f"Successfully created {stream_class.__name__} @ {sample_rate}Hz")
            print(f"✅ Successfully created {stream_class.__name__} @ {sample_rate}Hz")
            print(f"✅ Stream mode confirmed: {'HFP' if is_hfp_stream else 'Non-HFP'}")
            return stream
        except Exception as e:
            # 如果全双工失败，尝试仅输入模式
            if stream_class == sd.Stream and input_idx is not None:
                logging.warning(f"Full-duplex failed, trying input-only: {e}")
                print(f"⚠️ Full-duplex failed, falling back to input-only mode")
                kwargs['device'] = input_idx
                stream = sd.InputStream(**kwargs)
                print(f"✅ Fallback: Successfully created InputStream @ {sample_rate}Hz")
                return stream
            else:
                print(f"❌ Failed to create audio stream: {e}")
                raise RuntimeError(f"Failed to create audio stream: {e}")

    def list_devices(self):
        """列出所有音频设备"""
        print("\n=== Audio Device List ===")
        print("Index  Audio API   Type    Rate     Name")
        print("-" * 70)
        
        # Step 1: 分组目标设备（包含device_name的设备）
        target_device_groups = {}
        # Step 2: 分组其他设备
        other_device_groups = {}
        
        for idx, dev in enumerate(self.devices):
            # 忽略ASIO设备
            if dev['hostapi'] == 3:  # ASIO = 3
                continue
            name = dev['name']
            
            # 检查是否包含目标设备名称
            if self.device_name.lower() in name.lower():
                if name not in target_device_groups:
                    target_device_groups[name] = []
                target_device_groups[name].append((idx, dev))
            else:
                if name not in other_device_groups:
                    other_device_groups[name] = []
                other_device_groups[name].append((idx, dev))
        
        # 显示目标设备组
        if target_device_groups:
            print(f"=== Target Devices (containing '{self.device_name}') ===")
            for name, group in sorted(target_device_groups.items()):
                # 使用第一个设备来检查是否为目标设备（同名设备应该有相同的目标状态）
                is_target = self._is_target_device(group[0][1])
                # 先按Type排序(IN在前)，再按API类型排序
                sorted_group = sorted(group, key=lambda x: (x[1]['max_input_channels'] == 0, x[1]['hostapi']))
                for idx, dev in sorted_group:
                    print(self._format_device_info(idx, dev, is_target))
                if len(group) > 1:  # 如果同一设备有多个接口，添加分隔行
                    print("  " + "-" * 68)
            print("-" * 70)
        
        # 显示其他设备组
        if other_device_groups:
            print("=== Other Devices ===")
            for name, group in sorted(other_device_groups.items()):
                # 其他设备不是目标设备
                is_target = False
                # 先按Type排序(IN在前)，再按API类型排序
                sorted_group = sorted(group, key=lambda x: (x[1]['max_input_channels'] == 0, x[1]['hostapi']))
                for idx, dev in sorted_group:
                    print(self._format_device_info(idx, dev, is_target))
                if len(group) > 1:  # 如果同一设备有多个接口，添加分隔行
                    print("  " + "-" * 68)
            print("-" * 70)
        
        print("* marks target device endpoints")

    def refresh_devices(self):
        """刷新设备列表"""
        self.devices = sd.query_devices()
        self._find_device_endpoints()

    def create_stream(self, input_enabled: bool = True, 
                 output_enabled: bool = True,
                 sample_rate: int = 16000,
                 force_hfp: bool = False) -> sd.Stream:
        """创建音频流，确保使用相同的音频API"""
        input_idx = self.input_idx if input_enabled else None
        output_idx = self.output_idx if output_enabled else None
        
        # 验证设备API兼容性
        if input_idx is not None and output_idx is not None:
            input_api = self.devices[input_idx]['hostapi']
            output_api = self.devices[output_idx]['hostapi']
            
            if input_api != output_api:
                input_api_name = self.HOSTAPI_NAMES.get(input_api, f'API{input_api}')
                output_api_name = self.HOSTAPI_NAMES.get(output_api, f'API{output_api}')
                logging.warning(f"API mismatch: Input({input_api_name}) vs Output({output_api_name}), using input-only mode")
                output_idx = None  # 强制使用仅输入模式
    
        return self._create_audio_stream(
            input_idx=input_idx,
            output_idx=output_idx,
            sample_rate=sample_rate,
            force_hfp=force_hfp
        )

    def record(self, duration: int = 5, filename: Optional[str] = None) -> np.ndarray:
        """录制音频"""
        if self.input_idx is None:
            raise RuntimeError(f"No input endpoint found for {self.device_name}")

        frames = []
        try:
            with self.create_stream(input_enabled=True, output_enabled=True) as stream:
                is_hfp = self._is_hfp_mode(self.input_idx)
                stream.start()
                last_keepalive = time.time()
                
                end_time = time.time() + duration
                while time.time() < end_time:
                    data, overflow = stream.read(self.chunk_size)
                    frames.append(data)
                    
                    # HFP keepalive
                    if is_hfp and (time.time() - last_keepalive) >= self.hfp_keepalive_interval:
                        self._send_keepalive(stream)
                        last_keepalive = time.time()
                    remaining_time = end_time - time.time()
                    print(f"\rRecording data from Mic ... (remaining: {remaining_time:.1f}s)", end='', flush=True)
                        
        except Exception as e:
            logging.warning(f"Full-duplex mode failed({e}), switching to input-only mode...")
            with self.create_stream(input_enabled=True, output_enabled=False) as stream:
                stream.start()
                end_time = time.time() + duration
                while time.time() < end_time:
                    data, overflow = stream.read(self.chunk_size)
                    frames.append(data)

        audio_data = np.concatenate(frames, axis=0)
        if filename:
            sf.write(filename, audio_data, int(stream.samplerate))
        return audio_data

    def play(self, audio_data: np.ndarray, sample_rate: int = None):
        """播放音频数据"""
        if self.output_idx is None:
            raise RuntimeError(f"No output endpoint found for {self.device_name}")
            
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]  # Convert to mono
        
        # Convert data type to match stream expectations
        if audio_data.dtype != np.int16:
            # Normalize float data to [-1, 1] range if needed
            if audio_data.dtype in [np.float32, np.float64]:
                # Clip to [-1, 1] range to prevent overflow
                audio_data = np.clip(audio_data, -1.0, 1.0)
                # Convert to int16
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                # For other integer types, convert directly
                audio_data = audio_data.astype(np.int16)
            
        with self.create_stream(input_enabled=False, 
                              output_enabled=True,
                              sample_rate=sample_rate or 16000) as stream:
            stream.start()
            pos = 0
            while pos < len(audio_data):
                chunk = audio_data[pos:pos + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                stream.write(chunk)
                pos += self.chunk_size

    def _send_keepalive(self, stream):
        """发送保活信号"""
        zeros = np.zeros((self.chunk_size, self.channels), dtype=np.int16)
        stream.write(zeros)

    def _is_hfp_mode(self, device_idx: int) -> bool:
        """检查设备是否为HFP模式"""
        return 'HFP' == self._get_device_mode(self.devices[device_idx])

def main():
    parser = argparse.ArgumentParser(description='Audio Device Manager for ESP-ADF-AUDIO devices')
    parser.add_argument('-l', '--list', action='store_true', 
                       help='List all available audio devices')
    parser.add_argument('-r', '--record', type=int, metavar='SECONDS',
                       help='Record audio for specified number of seconds')
    parser.add_argument('-p', '--play', type=str, metavar='PATH',
                       help='Play audio file from specified path')
    parser.add_argument('-d', '--device', type=str, default='ESP-ADF-AUD',
                       help='Target device name to search for (default: ESP-ADF-AUD)')
    
    args = parser.parse_args()
    
    try:
        # 创建音频管理器实例
        audio_manager = AudioDeviceManager(device_name=args.device)
        
        # 处理命令行参数
        if args.list:
            audio_manager.list_devices()
            return 0
            
        if args.record:
            duration = args.record
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"recording_{timestamp}.wav"
            print(f"开始录音({duration}秒)...")
            audio_data = audio_manager.record(duration=duration, filename=filename)
            print(f"录音完成，已保存到 {filename}")
            return 0
            
        if args.play:
            try:
                print(f"播放音频文件: {args.play}")
                audio_data, sample_rate = sf.read(args.play)
                audio_manager.play(audio_data, sample_rate)
                print("播放完成")
                return 0
            except Exception as e:
                print(f"播放文件失败: {e}")
                return 1
            
        # 如果没有指定任何参数，显示默认行为
        if not any([args.list, args.record, args.play]):
            print("ESP-ADF-AUDIO 设备管理器")
            print("使用 -h 查看帮助信息")
            print("\n默认操作: 列出设备并录制10秒音频")
            
            audio_manager.list_devices()
            
            # 录制音频
            duration = 10
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            filename = f"test_{timestamp}.wav"
            print(f"\n开始录音({duration}秒)...")
            audio_data = audio_manager.record(duration=duration, filename=filename)
            print(f"录音完成，已保存到 {filename}")
            
            # 播放刚录制的音频
            print("\n播放录音...")
            audio_manager.play(audio_data)
            print("播放完成")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        logging.error("Exception details:", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

