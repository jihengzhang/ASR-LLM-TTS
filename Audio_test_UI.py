#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio_test_UI.py - Simple UI to control FunASR VAD/KWS pipeline with wxPython
"""
import sys
import os
import re
import threading
from datetime import datetime
import traceback
import pyaudio
import wx
# import wx.lib.scrolledpanel as scrolled

# Set up matplotlib backend using try-except for robustness
try:
    # Import our custom matplotlib setup module first
    # Now we can safely import pyplot and other matplotlib modules
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Error importing matplotlib: {e}")
    MATPLOTLIB_AVAILABLE = False

# Import processor after matplotlib setup
from FunASR_VAD_KWS_plot import VADKWSProcessor

class AudioTestFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None, title="Audio Test UI - FunASR VAD/KWS", size=(1280, 800))
        
        # Create main panel and sizer first
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Then initialize processor (moved down to avoid backend conflicts)
        self.processor = VADKWSProcessor(
            sample_rate=16000,
            chunk_size=8000,
            threshold=0.02,
            channels=1,
            silence_duration=0.3,
            buffer_duration=5.0,
            time_to_end_conversation=3,
            keywords=["hello", "Hi panda", "hi siri", "你好"],
            stopwords=["stop", "停止", "okay", "好了", "行了","好的", "退出"]

        )
        self.processor_started = False
        self.processor.isDebug = False
        # self.processor.isDebug = True
        
        # Set initial result message with F5 hint
        self.initial_message = "Audio Test UI started\n"
        self.initial_message += "Press F5 or click the Refresh button to update the list of available audio devices.\n"
        self.initial_message += "Select an audio input device from the dropdown menu.\n"
        
        # Create buttons
        button_size = wx.Size(35, 30)  # 约等于10个字符宽，高度增加一倍至60像素
        # Top frame for buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Create combined row for device selection and recording prompt
        top_row_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left side: device selection
        device_sizer = wx.BoxSizer(wx.HORIZONTAL)
        device_label = wx.StaticText(self.panel, label="Audio Input Device:")
        self.device_combobox = wx.Choice(self.panel, size=(300, -1))
        self.refresh_btn = wx.Button(self.panel, label="↻ Refresh", size=(90, 30))
        
        # Populate the device combobox
        self.populate_audio_devices()
        
        device_sizer.Add(device_label, 0, wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, 5)
        device_sizer.Add(self.device_combobox, 1, wx.EXPAND|wx.RIGHT, 5)
        device_sizer.Add(self.refresh_btn, 0)
        
        # Right side: recording prompt
        prompt_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.prompt_text = wx.StaticText(self.panel, label="按住空格键开始进行录音 (Hold SPACE to record)")
        font = self.prompt_text.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.prompt_text.SetFont(font)
        self.prompt_text.SetForegroundColour(wx.Colour(0, 0, 150))  # Dark blue color
        
        # Add recording status indicator
        self.recording_indicator = wx.StaticText(self.panel, label="⚫")
        self.recording_indicator.SetFont(font)
        self.recording_indicator.SetForegroundColour(wx.Colour(128, 128, 128))  # Gray when not recording
        
        prompt_sizer.Add(self.prompt_text, 0, wx.ALIGN_CENTER_VERTICAL)
        prompt_sizer.Add((20, -1), 0, wx.ALIGN_CENTER_VERTICAL)  # Spacer
        prompt_sizer.Add(self.recording_indicator, 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Add device selection and recording prompt to the combined row
        top_row_sizer.Add(device_sizer, 1, wx.ALIGN_CENTER_VERTICAL)
        top_row_sizer.AddStretchSpacer(1)  # Flexible space in between
        top_row_sizer.Add(prompt_sizer, 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Bind device selection event and refresh button
        self.device_combobox.Bind(wx.EVT_CHOICE, self.on_device_selected)
        self.refresh_btn.Bind(wx.EVT_BUTTON, self.on_refresh_devices)
        
   
        # 使用特殊样式创建按钮，禁用空格键触发和TAB焦点
        button_style = wx.BORDER_NONE  # 不使用WANTS_CHARS，因为它会捕获空格键
        
        self.start_btn = wx.Button(self.panel, label="Start VAD/KWS Processor", size=button_size, style=button_style)
        self.stop_btn = wx.Button(self.panel, label="Stop Processor", size=button_size, style=button_style)
        self.quit_btn = wx.Button(self.panel, label="Quit", size=button_size, style=button_style)
        
        # 禁用按钮的TAB焦点和空格键激活
        for btn in [self.start_btn, self.stop_btn, self.quit_btn]:
            # 设置样式标志
            btn.SetWindowStyleFlag(btn.GetWindowStyleFlag() | wx.NO_BORDER | wx.WANTS_CHARS)
            # 绑定空格键事件来阻止默认行为
            btn.Bind(wx.EVT_KEY_DOWN, self.on_button_key_down)
        
        # 设置面板接收所有键盘事件
        self.panel.SetWindowStyle(self.panel.GetWindowStyle() | wx.WANTS_CHARS)
        
        self.stop_btn.Disable()
        
        # Bind button events
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.quit_btn.Bind(wx.EVT_BUTTON, self.on_quit)
        
        # Add buttons to sizer with equal width
        btn_sizer.Add(self.start_btn, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        btn_sizer.Add(self.stop_btn, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        btn_sizer.Add(self.quit_btn, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        
        # Add the combined row and buttons to main sizer
        main_sizer.Add(top_row_sizer, proportion=0, flag=wx.EXPAND|wx.ALL, border=5)
        main_sizer.Add(btn_sizer, proportion=0, flag=wx.EXPAND)
        
        # Frame for matplotlib plot
        self.plot_panel = wx.Panel(self.panel)
        self.plot_panel.SetMinSize((600, 400))
        
        # Embed matplotlib figure in wxPython
        self.canvas = None
        if hasattr(self.processor, 'fig'):
            self.canvas = FigureCanvas(self.plot_panel, -1, self.processor.fig)
            plot_sizer = wx.BoxSizer(wx.VERTICAL)
            plot_sizer.Add(self.canvas, 1, wx.EXPAND|wx.ALL)
            self.plot_panel.SetSizer(plot_sizer)
            # 确保canvas获取合适的大小
            self.canvas.SetMinSize(self.plot_panel.GetMinSize())
        
        main_sizer.Add(self.plot_panel, proportion=8, flag=wx.EXPAND|wx.ALL, border=5)
        
        # Text box for KWS/ASR results with label at the top
        result_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Add title at the top, spanning full width
        result_label = wx.StaticText(self.panel, label="KWS/ASR Results:", 
                                    style=wx.ALIGN_LEFT)
        result_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        
        self.result_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.result_text.SetFont(wx.Font(11, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        # Set initial message with timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.result_text.SetValue(f"{timestamp}: {self.initial_message}")
        
        result_sizer.Add(result_label, proportion=0, flag=wx.EXPAND|wx.BOTTOM, border=5)
        result_sizer.Add(self.result_text, proportion=1, flag=wx.EXPAND)
        
        # Reduced text box height by 1/3 (from proportion=2 to proportion=1)
        main_sizer.Add(result_sizer, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        
        # Add status bar at the bottom of the UI
        status_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.status_text = wx.StaticText(self.panel, label="Status: Waiting for keyword", 
                                      style=wx.ALIGN_LEFT)
        self.status_text.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.status_text.SetForegroundColour(wx.Colour(0, 0, 150))  # Dark blue color
        
        status_sizer.Add(self.status_text, proportion=1, flag=wx.EXPAND|wx.LEFT, border=5)
        
        # Add status bar to main sizer
        main_sizer.Add(status_sizer, proportion=0, flag=wx.EXPAND|wx.BOTTOM|wx.LEFT|wx.RIGHT, border=5)
        
        self.panel.SetSizer(main_sizer)
        
        # Setup timers for updating the plot and results
        self.plot_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_plot_canvas, self.plot_timer)
        self.plot_timer.Start(100)  # Update every 100ms
        
        self.results_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update_results, self.results_timer)
        self.results_timer.Start(1000)  # Update every 100ms
        
        # Handle window close
        self.Bind(wx.EVT_CLOSE, self.on_close)
        
        # Add keyboard accelerator for F5 to refresh devices
        self.accel_table = wx.AcceleratorTable([
            (wx.ACCEL_NORMAL, wx.WXK_F5, wx.ID_REFRESH)
        ])
        self.SetAcceleratorTable(self.accel_table)
        self.Bind(wx.EVT_MENU, self.on_refresh_devices, id=wx.ID_REFRESH)
        
        # Add keyboard event handlers for space bar recording
        self.Bind(wx.EVT_KEY_DOWN, self.on_frame_key_down)
        self.Bind(wx.EVT_KEY_UP, self.on_frame_key_up)
        
        # Important: Also bind keyboard events directly to the panel
        self.panel.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.panel.Bind(wx.EVT_KEY_UP, self.on_key_up)
        
        # Make the panel able to receive keyboard focus
        self.panel.SetFocusIgnoringChildren()
        
        # Recording state variables
        self.space_pressed = False
        self.is_recording = False
        self.recording_start_time = None
        self.recording_frames = []
        self.recording_sample_rate = 16000  # Make sure this matches processor's sample rate
        self.recordings_dir = "recordings"  # Directory to store recordings
        
        # Ensure recording directory exists
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        
    def on_start(self, event):
        # if self.processor_started:
        #     wx.MessageBox("Processor already running.", "Info", wx.OK | wx.ICON_INFORMATION)
        #     return
        threading.Thread(target=self.processor.start, daemon=True).start()
        self.processor_started = True
        self.stop_btn.Enable()
        self.start_btn.Disable()
        
        # 将焦点设置到面板上，而不是按钮
        wx.CallAfter(self.panel.SetFocusIgnoringChildren)  # 使用CallAfter确保在UI更新完成后设置焦点并忽略子控件
        # 确保面板能够捕获按键事件
        wx.CallAfter(self.panel.SetFocus)

        # 检查是否有新的fig，必要时重建canvas
        # if hasattr(self.processor, 'fig'):
        #     if self.canvas is None or self.canvas.figure != self.processor.fig:
        #         # 先销毁旧canvas
        #         if self.canvas:
        #             self.canvas.Destroy()
        #         # 创建新canvas
        #         self.canvas = FigureCanvas(self.plot_panel, -1, self.processor.fig)
        #         plot_sizer = wx.BoxSizer(wx.VERTICAL)
        #         plot_sizer.Add(self.canvas, 1, wx.EXPAND|wx.ALL)
        #         self.plot_panel.SetSizer(plot_sizer)
        #         self.canvas.SetMinSize(self.plot_panel.GetMinSize())
        #         self.plot_panel.Layout()

        # # 重新启动 plot 和结果刷新
        # if not self.plot_timer.IsRunning():
        #     self.plot_timer.Start(100)
        if not self.results_timer.IsRunning():
            self.results_timer.Start(1000)
    
    def on_stop(self, event):
        if self.processor and self.processor_started:
            self.processor.stop()
            self.processor_started = False
            self.plot_timer.Stop()
            self.results_timer.Stop()
            self.stop_btn.Disable()
            self.start_btn.Enable()
    
    def on_quit(self, event):
        self.on_stop(event)
        self.Close()
    
    def update_plot_canvas(self, event):
        try:
            if self.canvas and hasattr(self.processor, 'fig'):
                # 使用CallAfter确保在主线程中更新UI
                wx.CallAfter(self.canvas.draw)
                
                # 如果处理器有更新图表数据的方法，则调用它
                if hasattr(self.processor, 'update_plot_data') and callable(self.processor.update_plot_data):
                    self.processor.update_plot_data()
        except Exception as e:
            print(f"Error updating plot: {e}")
    
    def update_results(self, event):
        if self.processor and hasattr(self.processor, 'detected_keywords'):
            text = ""
            # 获取最近10条关键词检测结果
            for kw, ts in list(self.processor.detected_keywords)[-10:]:
                # 格式化时间戳
                if isinstance(ts, float):
                    ts_str = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                else:
                    ts_str = str(ts)
                text += f"{ts_str}: {kw}\n"
            
            # 更新文本框并滚动到底部
            self.result_text.SetValue(text)
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
            
            # Update status text at the bottom of UI
            if hasattr(self.processor, 'status'):
                self.status_text.SetLabel(f"Status: {self.processor.status}")
    
    def populate_audio_devices(self):
        """Populate the audio devices dropdown"""
        try:
            # Get list of audio devices from processor
            if self.processor:
                devices, default_device = self.processor.get_audio_devices()
                
                # Store device indices and names
                self.audio_devices = devices
                default_device_index = -1
                
                # Clear existing items
                self.device_combobox.Clear()
                
                # Add devices to combobox
                for i, (device_id, name) in enumerate(devices):
                    self.device_combobox.Append(f"{name} (Device {device_id})")
                    
                    # Track the default device's position in the list
                    if default_device and device_id == default_device[0]:
                        default_device_index = i
                
                # Select default device
                if default_device_index >= 0:
                    self.device_combobox.SetSelection(default_device_index)
                elif len(devices) > 0:
                    self.device_combobox.SetSelection(0)
                    
                # Ensure the processor uses the selected device
                if len(devices) > 0:
                    selected_index = default_device_index if default_device_index >= 0 else 0
                    device_index = devices[selected_index][0]
                    self.processor.input_device_index = device_index
                    print(f"Set audio device to {devices[selected_index][1]} (index {device_index})")
            else:
                print("Processor not initialized yet, can't get audio devices")
                
        except Exception as e:
            print(f"Error populating audio devices: {e}")
            traceback.print_exc()
    
    def on_device_selected(self, event):
        """Handle device selection from dropdown"""
        try:
            # Temporarily disable UI components during device change
            self.device_combobox.Disable()
            self.refresh_btn.Disable()
            
            # Update device selection visually
            selection = self.device_combobox.GetSelection()
            if selection != wx.NOT_FOUND and selection < len(self.audio_devices):
                device_index = self.audio_devices[selection][0]
                device_name = self.audio_devices[selection][1]
                
                print(f"Selected audio device: {device_name} (index {device_index})")
                
                # Change button color to show processing
                self.refresh_btn.SetBackgroundColour(wx.Colour(255, 255, 0))  # Yellow
                self.refresh_btn.SetLabel("Changing...")
                
                # Force UI update immediately
                wx.Yield()
                
                # Update the result text to show the selected device
                current_text = self.result_text.GetValue()
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.result_text.SetValue(f"{current_text}\n{timestamp}: Changed device to {device_name}")
                self.result_text.ShowPosition(self.result_text.GetLastPosition())
                
                # Update the processor's input device
                if self.processor:
                    self.processor.input_device_index = device_index
                    
                    # If processor is running, restart it to apply the new device
                    if self.processor_started:
                        message = f"To use the new audio device '{device_name}', the processor will be restarted."
                        wx.MessageBox(message, "Device Changed", wx.OK | wx.ICON_INFORMATION)
                        self.on_stop(None)
                        self.on_start(None)
                        
                        # Add status message to results
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        current_text = self.result_text.GetValue()
                        self.result_text.SetValue(f"{current_text}\n{timestamp}: Processor restarted with new device")
                        self.result_text.ShowPosition(self.result_text.GetLastPosition())
                
                # Reset button appearance when done
                self.refresh_btn.SetBackgroundColour(wx.NullColour)
                self.refresh_btn.SetLabel("\u21bb Refresh")
                
        except Exception as e:
            print(f"Error setting audio device: {e}")
            traceback.print_exc()
            
            # Show error in results
            timestamp = datetime.now().strftime('%H:%M:%S')
            current_text = self.result_text.GetValue()
            self.result_text.SetValue(f"{current_text}\n{timestamp}: ERROR: Failed to change audio device - {str(e)}")
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
            
            # Reset button appearance on error
            self.refresh_btn.SetBackgroundColour(wx.Colour(255, 200, 200))  # Light red
            self.refresh_btn.SetLabel("Error")
            
        finally:
            # Re-enable UI components
            self.device_combobox.Enable()
            self.refresh_btn.Enable()
    
    def on_refresh_devices(self, event):
        """Refresh the list of audio devices"""
        try:
            # Temporarily disable UI components during refresh
            self.device_combobox.Disable()
            self.refresh_btn.Disable()
            self.refresh_btn.SetBackgroundColour(wx.Colour(173, 216, 230))  # Light blue
            self.refresh_btn.SetLabel("Refreshing...")
            
            # Force UI update immediately
            wx.Yield()
            
            print("Refreshing audio devices list...")
            
            # Remember the previously selected device name if any
            prev_selection = self.device_combobox.GetSelection()
            prev_device_name = None
            if prev_selection != wx.NOT_FOUND and prev_selection < len(self.audio_devices):
                prev_device_name = self.audio_devices[prev_selection][1]
            
            # Refresh the device list
            self.populate_audio_devices()
            
            # Try to select the previously selected device by name
            if prev_device_name:
                for i, (_, name) in enumerate(self.audio_devices):
                    if name == prev_device_name:
                        self.device_combobox.SetSelection(i)
                        break
            
            # Add status message to results
            timestamp = datetime.now().strftime('%H:%M:%S')
            current_text = self.result_text.GetValue()
            self.result_text.SetValue(f"{current_text}\n{timestamp}: Audio devices list refreshed")
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
            
            # Show success visual feedback
            self.refresh_btn.SetBackgroundColour(wx.Colour(144, 238, 144))  # Light green
            self.refresh_btn.SetLabel("✓ Updated")
            
            # Schedule reset of button appearance after 1 second
            wx.CallLater(1000, self.reset_refresh_button)
            
        except Exception as e:
            print(f"Error refreshing audio devices: {e}")
            traceback.print_exc()
            
            # Show error in results
            timestamp = datetime.now().strftime('%H:%M:%S')
            current_text = self.result_text.GetValue()
            self.result_text.SetValue(f"{current_text}\n{timestamp}: ERROR: Failed to refresh audio devices - {str(e)}")
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
            
            # Show error visual feedback
            self.refresh_btn.SetBackgroundColour(wx.Colour(255, 200, 200))  # Light red
            self.refresh_btn.SetLabel("✗ Error")
            
            # Schedule reset of button appearance after 2 seconds
            wx.CallLater(2000, self.reset_refresh_button)
            
        finally:
            # Re-enable UI components
            self.device_combobox.Enable()
            self.refresh_btn.Enable()
    
    def reset_refresh_button(self):
        """Reset the refresh button to its original appearance"""
        self.refresh_btn.SetBackgroundColour(wx.NullColour)
        self.refresh_btn.SetLabel("\u21bb Refresh")
    
    def on_key_down(self, event):
        """Handle key down event for space bar recording"""
        # 记录按键动作以便调试
        key_code = event.GetKeyCode()
        
        # 阻止空格键的默认行为（激活按钮）
        if key_code == wx.WXK_SPACE:
            # 立即阻止事件继续传播，防止按钮激活
            if not self.space_pressed:
                # 只有当处理器已启动时才开始录音
                if not self.processor_started:
                    wx.MessageBox("Please start the VAD/KWS processor first", "Warning", 
                                 wx.OK | wx.ICON_INFORMATION)
                else:
                    self.space_pressed = True
                    # 使用处理器的录音功能
                    if self.processor.start_recording():
                        # 更改录音指示器颜色为红色
                        self.recording_indicator.SetForegroundColour(wx.Colour(255, 0, 0))
                        # 更新状态文本为 "Recording"
                        self.status_text.SetLabel("Status: Recording")
                        self.recording_indicator.SetLabel("⚫ Recording")
                        
                        # 更新UI以反映录音状态 - 使用简单的信息
                        timestamp = datetime.now().strftime('%H:%M:%S')
                        current_text = self.result_text.GetValue()
                        self.result_text.SetValue(f"{current_text}\n{timestamp}: 正在录音 (now it's recording...)")
                        self.result_text.ShowPosition(self.result_text.GetLastPosition())
                    else:
                        # 录音启动失败，记录错误信息
                        self.space_pressed = False
                        self.result_text.SetValue(f"{current_text}\n{timestamp}: Failed to start recording")
                        self.result_text.ShowPosition(self.result_text.GetLastPosition())
            # 无论如何都不传播空格键事件
            return
        # 其他键正常处理
        event.Skip()
    
    def on_frame_key_down(self, event):
        """Frame-level key down handler - 转发所有事件到panel"""
        # 将所有键盘事件转发到面板以确保一致处理
        if event.GetKeyCode() == wx.WXK_SPACE:
            # 阻止空格键的默认按钮行为
            self.panel.SetFocusIgnoringChildren()  # 确保焦点在面板上且忽略子控件
            
            # 创建一个新的键盘事件并发送到面板，这样事件就能被面板处理
            new_event = wx.KeyEvent(wx.wxEVT_KEY_DOWN)
            new_event.m_keyCode = wx.WXK_SPACE
            wx.PostEvent(self.panel, new_event)
            return
        event.Skip()
    
    def on_frame_key_up(self, event):
        """Frame-level key up handler - 转发所有事件到panel"""
        # 将所有键盘事件转发到面板以确保一致处理
        if event.GetKeyCode() == wx.WXK_SPACE:
            self.panel.SetFocus()  # 确保焦点在面板上
            self.on_key_up(event)  # 直接调用面板的按键处理
            return
        event.Skip()
    
    def on_key_up(self, event):
        """Handle key up event for space bar recording"""
        if event.GetKeyCode() == wx.WXK_SPACE and self.space_pressed:
            self.space_pressed = False
            
            # 停止处理器中的录音
            self.processor.stop_recording()
            
            # 恢复录音指示器颜色为灰色
            self.recording_indicator.SetForegroundColour(wx.Colour(128, 128, 128))
            self.recording_indicator.SetLabel("⚫")
            
            # 恢复状态文本为等待关键词
            self.status_text.SetLabel("Status: Waiting for keyword")
            
            # 简单显示录音已停止
            timestamp = datetime.now().strftime('%H:%M:%S')
            current_text = self.result_text.GetValue()
            self.result_text.SetValue(f"{current_text}\n{timestamp}: 录音已停止")
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
            
            return  # 阻止事件继续传播
            
        event.Skip()
        
    def on_recording_saved(self, filename):
        """录音保存完成后的回调函数
        
        Args:
            filename: 保存的文件路径
        """
        # 使用wx.CallAfter确保在主线程中更新UI
        def update_ui():
            timestamp = datetime.now().strftime('%H:%M:%S')
            current_text = self.result_text.GetValue()
            if filename:
                self.result_text.SetValue(f"{current_text}\n{timestamp}: 录音已保存为 '{os.path.basename(filename)}'")
            else:
                self.result_text.SetValue(f"{current_text}\n{timestamp}: 没有录音数据可保存")
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
        wx.CallAfter(update_ui)
    
    # 注意：录音功能现在由 VADKWSProcessor 类实现
    # 旧的录音方法已移除，现在使用 processor.start_recording() 和 processor.save_recording()
    
    def on_button_key_down(self, event):
        """处理按钮上的键盘事件，防止空格键激活按钮"""
        # 如果是空格键，完全阻止事件并返回
        if event.GetKeyCode() == wx.WXK_SPACE:
            # 将焦点设回面板
            self.panel.SetFocusIgnoringChildren()
            
            # 记录调试信息
            timestamp = datetime.now().strftime('%H:%M:%S')
            current_text = self.result_text.GetValue()
            self.result_text.SetValue(f"{current_text}\n{timestamp}: Space intercepted from button")
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
            
            # 转发空格键事件到面板的处理器
            new_event = wx.KeyEvent(wx.wxEVT_KEY_DOWN)
            new_event.m_keyCode = wx.WXK_SPACE
            wx.PostEvent(self.panel, new_event)
            return
        # 其他键正常处理
        event.Skip()
    
    def on_close(self, event):
        # 停止处理器和定时器
        if self.processor and self.processor_started:
            self.processor.stop()

        self.plot_timer.Stop()
        self.results_timer.Stop()

        # 安全停止matplotlib动画
        animation = getattr(self.processor, "animation", None)
        if animation and getattr(animation, "event_source", None):
            animation.event_source.stop()
            animation._fig = None  # 解除对figure的引用

        if self.canvas:
            self.canvas.Destroy()

        self.Destroy()
        # 强制退出主进程
        wx.CallAfter(sys.exit)
def run_processor_ui():
    app = wx.App(False)
    frame = AudioTestFrame()
    frame.Show()
    app.MainLoop()

if __name__ == "__main__":
    run_processor_ui()
