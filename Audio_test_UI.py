#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio_test_UI.py - Simple UI to control FunASR VAD/KWS pipeline
"""
import sys
import threading
import tkinter as tk
from tkinter import messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from FunASR_VAD_KWS_plot import VADKWSProcessor

def run_processor_ui():
    processor = None
    processor_started = False

    # Initialize processor and plot at startup
    processor = VADKWSProcessor(
        sample_rate=16000,
        chunk_size=1600,
        threshold=0.01,
        silence_duration=2.0,
        buffer_duration=3.0,
        keywords=["hello", "computer", "system"]
    )
    processor_started = False

    def start_processor():
        nonlocal processor_started
        if processor_started:
            messagebox.showinfo("Info", "Processor already running.")
            return
        threading.Thread(target=processor.start, daemon=True).start()
        processor_started = True

    def stop_processor():
        nonlocal processor, processor_started
        if processor and processor_started:
            processor.stop()
            processor_started = False
            # messagebox.showinfo("Info", "Processor stopped.")
        else:
            pass
            # messagebox.showinfo("Info", "Processor not running.")

    root = tk.Tk()
    root.title("Audio Test UI - FunASR VAD/KWS")
    root.geometry("1920x1080")

    # Use grid layout for auto-resizing
    root.grid_rowconfigure(1, weight=8)  # plot row
    root.grid_rowconfigure(2, weight=2)  # text row
    root.grid_columnconfigure(0, weight=1)

    # Top frame for buttons
    top_frame = tk.Frame(root)
    top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
    top_frame.grid_columnconfigure(0, weight=1)
    top_frame.grid_columnconfigure(1, weight=1)
    top_frame.grid_columnconfigure(2, weight=1)

    start_btn = tk.Button(top_frame, text="Start VAD/KWS Processor", command=start_processor, width=12, height=1)
    start_btn.grid(row=0, column=0, padx=10, sticky="ew")

    stop_btn = tk.Button(top_frame, text="Stop Processor", command=stop_processor, width=12, height=1)
    stop_btn.grid(row=0, column=1, padx=10, sticky="ew")

    quit_btn = tk.Button(top_frame, text="Quit", command=root.quit, width=12, height=1)
    quit_btn.grid(row=0, column=2, padx=10, sticky="ew")

    # Frame for matplotlib plot
    plot_frame = tk.Frame(root)
    plot_frame.grid(row=1, column=0, sticky="nsew")
    plot_frame.grid_rowconfigure(0, weight=1)
    plot_frame.grid_columnconfigure(0, weight=1)

    # Text box for KWS/ASR results
    result_frame = tk.Frame(root)
    result_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
    result_frame.grid_rowconfigure(0, weight=1)
    result_frame.grid_columnconfigure(1, weight=1)
    result_label = tk.Label(result_frame, text="KWS/ASR Results:", font=("Arial", 12))
    result_label.grid(row=0, column=0, sticky="nw")
    result_text = tk.Text(result_frame, height=6, font=("Consolas", 11))
    result_text.grid(row=0, column=1, sticky="nsew")
    result_text.config(state=tk.DISABLED)

    # Embed matplotlib figure in Tkinter immediately
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    canvas = None
    if processor and hasattr(processor, 'fig'):
        canvas = FigureCanvasTkAgg(processor.fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot_canvas():
        if canvas:
            canvas.draw()
        root.after(100, update_plot_canvas)  # Redraw every 100ms

    update_plot_canvas()  # Start canvas redraw loop

    def update_results():
        nonlocal processor
        if processor and hasattr(processor, 'detected_keywords'):
            result_text.config(state=tk.NORMAL)
            result_text.delete(1.0, tk.END)
            for kw, ts in processor.detected_keywords[-10:]:
                result_text.insert(tk.END, f"{ts.strftime('%H:%M:%S')}: {kw}\n")
            result_text.config(state=tk.DISABLED)
        root.after(1000, update_results)  # Update every second

    update_results()  # Start updating results at startup

    # Patch processor.start to refresh plot after starting (if needed)
    orig_start = start_processor
    def patched_start():
        orig_start()
        # Optionally refresh plot if processor.fig is recreated
        # (not strictly needed since fig is created at init)
    start_btn.config(command=patched_start)

    root.mainloop()

if __name__ == "__main__":
    run_processor_ui()
