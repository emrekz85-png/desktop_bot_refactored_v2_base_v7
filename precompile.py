#!/usr/bin/env python3
"""
PRECOMPILE SCRIPT
=================
Kütüphaneleri önceden derleyerek başlatma süresini azaltır.
İlk kurulumdan sonra bir kez çalıştırın.

Kullanım: python precompile.py
"""

import sys
import py_compile
import compileall
import time
import os

def precompile():
    print("=" * 50)
    print("  Trading Bot - Precompile Tool")
    print("=" * 50)
    print()

    start = time.time()

    # 1. Ana script'i derle
    print("[1/4] Ana script derleniyor...")
    try:
        py_compile.compile("desktop_bot_refactored_v2_base_v7.py", doraise=True)
        print("      ✓ desktop_bot_refactored_v2_base_v7.py")
    except Exception as e:
        print(f"      ✗ Hata: {e}")

    # 2. fast_start.py derle
    print("[2/4] Fast launcher derleniyor...")
    try:
        py_compile.compile("fast_start.py", doraise=True)
        print("      ✓ fast_start.py")
    except Exception as e:
        print(f"      ✗ Hata: {e}")

    # 3. Kritik kütüphaneleri import et (cache'e al)
    print("[3/4] Kritik kütüphaneler yükleniyor (cache için)...")

    libs = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("requests", "Requests"),
        ("matplotlib", "Matplotlib"),
    ]

    for lib_name, display_name in libs:
        try:
            __import__(lib_name)
            print(f"      ✓ {display_name}")
        except ImportError:
            print(f"      ✗ {display_name} (yüklü değil)")

    # 4. PyQt5 import et
    print("[4/4] PyQt5 yükleniyor (cache için)...")
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QThread
        print("      ✓ PyQt5")
    except ImportError:
        print("      ✗ PyQt5 (yüklü değil)")

    elapsed = time.time() - start
    print()
    print("=" * 50)
    print(f"  Tamamlandı! ({elapsed:.1f} saniye)")
    print("  Sonraki başlatmalar daha hızlı olacak.")
    print("=" * 50)


if __name__ == "__main__":
    precompile()
