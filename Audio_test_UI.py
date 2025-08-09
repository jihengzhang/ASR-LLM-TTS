#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio_test_UI.py - Simple UI to control FunASR VAD/KWS pipeline with wxPython
"""
import sys
import threading
import datetime
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
            threshold=0.01,
            channels=1,
            silence_duration=0.4,
            buffer_duration=5.0,
            keywords=["hello", "Hi panda", "hi siri"]
        )
        self.processor_started = False
        self.processor.isDebug = False
        # self.processor.isDebug = True
        
        # Top frame for buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Create buttons
        button_size = wx.Size(35, -1)  # 约等于10个字符宽，70像素可根据实际调整
        self.start_btn = wx.Button(self.panel, label="Start VAD/KWS Processor", size=button_size)
        self.stop_btn = wx.Button(self.panel, label="Stop Processor", size=button_size)
        self.quit_btn = wx.Button(self.panel, label="Quit", size=button_size)
        self.stop_btn.Disable()
        
        # Bind button events
        self.start_btn.Bind(wx.EVT_BUTTON, self.on_start)
        self.stop_btn.Bind(wx.EVT_BUTTON, self.on_stop)
        self.quit_btn.Bind(wx.EVT_BUTTON, self.on_quit)
        
        # Add buttons to sizer with equal width
        btn_sizer.Add(self.start_btn, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)
        btn_sizer.Add(self.stop_btn, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)
        btn_sizer.Add(self.quit_btn, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)
        
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
        
        main_sizer.Add(self.plot_panel, proportion=8, flag=wx.EXPAND|wx.ALL, border=10)
        
        # Text box for KWS/ASR results
        result_sizer = wx.BoxSizer(wx.HORIZONTAL)
        result_label = wx.StaticText(self.panel, label="KWS/ASR Results:", 
                                    style=wx.ALIGN_LEFT)
        result_label.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        self.result_text = wx.TextCtrl(self.panel, style=wx.TE_MULTILINE|wx.TE_READONLY)
        self.result_text.SetFont(wx.Font(11, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        
        result_sizer.Add(result_label, proportion=0, flag=wx.ALIGN_TOP|wx.ALL, border=5)
        result_sizer.Add(self.result_text, proportion=1, flag=wx.EXPAND|wx.ALL, border=5)
        
        main_sizer.Add(result_sizer, proportion=2, flag=wx.EXPAND|wx.ALL, border=10)
        
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
        
    def on_start(self, event):
        # if self.processor_started:
        #     wx.MessageBox("Processor already running.", "Info", wx.OK | wx.ICON_INFORMATION)
        #     return
        threading.Thread(target=self.processor.start, daemon=True).start()
        self.processor_started = True
        self.stop_btn.Enable()
        self.start_btn.Disable()

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
                    ts_str = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                else:
                    ts_str = str(ts)
                text += f"{ts_str}: {kw}\n"
            
            # 更新文本框并滚动到底部
            self.result_text.SetValue(text)
            self.result_text.ShowPosition(self.result_text.GetLastPosition())
    
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
