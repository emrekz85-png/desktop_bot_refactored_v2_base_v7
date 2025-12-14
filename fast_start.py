#!/usr/bin/env python3
"""
FAST STARTUP LAUNCHER
=====================
Bu script botu hızlı başlatır:
1. Anında splash screen gösterir (tkinter - çok hızlı)
2. Arka planda ağır kütüphaneleri yükler
3. Yükleme bitince ana pencereyi açar

Kullanım: python fast_start.py
"""

import sys
import os
import threading
import time

# Splash screen için tkinter (Python ile birlikte gelir, anında yüklenir)
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False


class SplashScreen:
    """Hızlı splash screen - yükleme sırasında gösterilir."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trading Bot")
        self.root.overrideredirect(True)  # Çerçevesiz pencere

        # Ekran ortasına yerleştir
        width, height = 400, 200
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - width) // 2
        y = (screen_h - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        # Stil
        self.root.configure(bg="#121212")

        # Başlık
        title = tk.Label(
            self.root,
            text="🚀 Trading Bot v39.0",
            font=("Segoe UI", 18, "bold"),
            fg="#00ccff",
            bg="#121212"
        )
        title.pack(pady=(30, 10))

        # Durum mesajı
        self.status_label = tk.Label(
            self.root,
            text="Başlatılıyor...",
            font=("Segoe UI", 11),
            fg="#888888",
            bg="#121212"
        )
        self.status_label.pack(pady=5)

        # Progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(
            "Custom.Horizontal.TProgressbar",
            troughcolor='#333333',
            background='#00ccff',
            darkcolor='#00ccff',
            lightcolor='#00ccff',
            bordercolor='#121212'
        )

        self.progress = ttk.Progressbar(
            self.root,
            style="Custom.Horizontal.TProgressbar",
            length=300,
            mode='determinate'
        )
        self.progress.pack(pady=20)

        # Alt bilgi
        info = tk.Label(
            self.root,
            text="İlk başlatma biraz uzun sürebilir",
            font=("Segoe UI", 9),
            fg="#555555",
            bg="#121212"
        )
        info.pack(pady=5)

        self.root.update()

    def update_status(self, text, progress=None):
        """Durum mesajını ve progress bar'ı güncelle."""
        self.status_label.config(text=text)
        if progress is not None:
            self.progress['value'] = progress
        self.root.update()

    def close(self):
        """Splash screen'i kapat."""
        self.root.destroy()


def load_and_run():
    """Ana uygulamayı yükle ve çalıştır."""
    global splash, app_loaded, main_module

    try:
        # 1. Temel kütüphaneler
        if splash:
            splash.update_status("NumPy yükleniyor...", 10)
        import numpy

        if splash:
            splash.update_status("Pandas yükleniyor...", 25)
        import pandas

        # 2. PyQt5
        if splash:
            splash.update_status("PyQt5 yükleniyor...", 40)
        from PyQt5.QtWidgets import QApplication

        # 3. Ana modül
        if splash:
            splash.update_status("Trading Bot yükleniyor...", 60)

        # Ana modülü import et
        import desktop_bot_refactored_v2_base_v7 as main_module

        if splash:
            splash.update_status("Hazırlanıyor...", 90)

        # Splash'ı kapat
        if splash:
            splash.update_status("Başlatılıyor!", 100)
            time.sleep(0.3)
            splash.close()
            splash = None

        # Ana uygulamayı başlat
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        window = main_module.MainWindow()
        window.show()
        sys.exit(app.exec_())

    except Exception as e:
        if splash:
            splash.update_status(f"Hata: {e}", 0)
            time.sleep(3)
            splash.close()
        print(f"Hata: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# Global değişkenler
splash = None
app_loaded = False
main_module = None


def main():
    global splash

    print("=" * 50)
    print("  Trading Bot v39.0 - Fast Launcher")
    print("=" * 50)

    if HAS_TK:
        # Splash screen oluştur
        splash = SplashScreen()
        splash.update_status("Kütüphaneler yükleniyor...", 5)

        # Yüklemeyi splash event loop'unda yap
        splash.root.after(100, load_and_run)
        splash.root.mainloop()
    else:
        # tkinter yoksa direkt yükle
        print("Yükleniyor... (bu biraz sürebilir)")
        load_and_run()


if __name__ == "__main__":
    main()
