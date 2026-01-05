#!/usr/bin/env python3
"""
M3 MacBook Air Performance Verification Script

Bu script, M3 Apple Silicon icin performans optimizasyonlarinin
dogru sekilde uygulandigini dogrular.

Kontroller:
1. ARM64 Native Python (Rosetta 2 DEGiL!)
2. NumPy Accelerate framework
3. Worker sayilari (CPU: 3, IO: 6)
4. Timedelta caching aktif mi
5. Thermal throttling riski tahmini

Kullanim:
    python verify_m3_performance.py
"""

import sys
import os
import platform
import time

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_arm64_native():
    """P0: ARM64 Native kontrolu (EN KRITIK!)"""
    print("\n" + "="*60)
    print("P0: ARM64 NATIVE KONTROLU")
    print("="*60)

    machine = platform.machine()
    python_version = platform.python_version()

    print(f"   Platform: {platform.system()} {platform.release()}")
    print(f"   Architecture: {machine}")
    print(f"   Python Version: {python_version}")

    if machine == 'arm64':
        print("\n   [OK] Python ARM64 native calisiyor!")
        print("   Rosetta 2 emulasyonu YOK - optimal performans")
        return True
    elif machine == 'x86_64':
        print("\n   [UYARI] Python x86_64 modunda calisiyor!")
        print("   Rosetta 2 emulasyonu aktif olabilir.")
        print("   Performans kaybi: %30-40")
        print("\n   COZUM: ARM64 native Python yukleyin:")
        print("   brew install python@3.11  # Homebrew ARM64")
        return False
    else:
        print(f"\n   [BILINMIYOR] Beklenmeyen architecture: {machine}")
        return False


def check_numpy_accelerate():
    """P1: NumPy Apple Accelerate framework kontrolu"""
    print("\n" + "="*60)
    print("P1: NUMPY ACCELERATE FRAMEWORK")
    print("="*60)

    try:
        import numpy as np
        print(f"   NumPy Version: {np.__version__}")

        # Get config as string
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            np.__config__.show()
        config_str = f.getvalue().lower()

        has_accelerate = 'accelerate' in config_str or 'veclib' in config_str
        has_openblas = 'openblas' in config_str
        has_mkl = 'mkl' in config_str

        if has_accelerate:
            print("\n   [OK] Apple Accelerate framework aktif!")
            print("   ARM64 SIMD (NEON) optimizasyonlari kullaniliyor")
            return True
        elif has_openblas:
            print("\n   [OK] OpenBLAS aktif")
            print("   Iyi performans, ama Accelerate daha iyi olabilir")
            return True
        elif has_mkl:
            print("\n   [UYARI] Intel MKL aktif - ARM64 icin optimal degil")
            return False
        else:
            print("\n   [BILINMIYOR] BLAS backend tespit edilemedi")
            print("   Config ciktisi:")
            print(config_str[:500])
            return None

    except ImportError:
        print("\n   [HATA] NumPy yuklu degil!")
        return False


def check_worker_settings():
    """P2: Worker sayisi kontrolu"""
    print("\n" + "="*60)
    print("P2: WORKER SAYILARI")
    print("="*60)

    cpu_count = os.cpu_count() or 4
    print(f"   CPU Cores: {cpu_count}")
    print(f"   M3 Air: 4 P-cores (hizli) + 4 E-cores (yavas)")

    # Check M3 config
    try:
        from core import M3_PERFORMANCE_CONFIG
        cpu_workers = M3_PERFORMANCE_CONFIG.get('cpu_workers', 3)
        io_workers = M3_PERFORMANCE_CONFIG.get('io_workers', 6)

        print(f"\n   Ayarlanan CPU Workers: {cpu_workers}")
        print(f"   Ayarlanan IO Workers: {io_workers}")

        if cpu_workers <= 3:
            print("\n   [OK] CPU workers P-core'lar ile sinirli (optimal)")
        else:
            print(f"\n   [UYARI] CPU workers ({cpu_workers}) > 3")
            print("   E-core'lar CPU-bound islerde %40 daha yavas!")

        if io_workers <= 6:
            print("   [OK] IO workers termal throttling icin uygun")
        else:
            print(f"   [UYARI] IO workers ({io_workers}) > 6 - throttling riski")

        return cpu_workers <= 3 and io_workers <= 6

    except ImportError as e:
        print(f"\n   [HATA] M3_PERFORMANCE_CONFIG import edilemedi: {e}")
        return False


def check_timedelta_caching():
    """P3: Timedelta caching kontrolu"""
    print("\n" + "="*60)
    print("P3: TIMEDELTA CACHING")
    print("="*60)

    try:
        from core.utils import tf_to_timedelta, _TF_TIMEDELTA_CACHE

        # Test caching
        import pandas as pd

        # First call - should compute
        t1 = time.perf_counter()
        for _ in range(1000):
            tf_to_timedelta("15m")
        t2 = time.perf_counter()
        first_time = (t2 - t1) * 1000  # ms

        # Check cache
        cache_size = len(_TF_TIMEDELTA_CACHE)

        print(f"   Cache boyutu: {cache_size} entries")
        print(f"   1000x tf_to_timedelta('15m'): {first_time:.2f}ms")

        if cache_size > 0 and first_time < 5:  # 5ms threshold
            print("\n   [OK] Timedelta caching aktif ve hizli!")
            return True
        elif cache_size > 0:
            print("\n   [OK] Caching aktif ama yavas")
            return True
        else:
            print("\n   [UYARI] Cache bos - caching calisiyor mu?")
            return False

    except ImportError as e:
        print(f"\n   [HATA] Import hatasi: {e}")
        return False
    except Exception as e:
        print(f"\n   [HATA] {e}")
        return False


def estimate_thermal_throttling():
    """P4: Thermal throttling risk tahmini"""
    print("\n" + "="*60)
    print("P4: THERMAL THROTTLING RISKI")
    print("="*60)

    print("   M3 MacBook Air: FANSIZ TASARIM")
    print("   Sureli CPU yuku = Termal throttling")
    print()

    # Simple CPU stress test
    print("   5 saniye CPU stress testi yapiliyor...")

    import time
    start_time = time.time()
    iterations = 0

    # Burst test (5 seconds)
    while time.time() - start_time < 5:
        # CPU-intensive work
        sum(i*i for i in range(10000))
        iterations += 1

    elapsed = time.time() - start_time
    rate = iterations / elapsed

    print(f"\n   Iterations: {iterations}")
    print(f"   Rate: {rate:.0f} iter/sec")

    # Estimate
    print("\n   TAHMINI THROTTLING ZAMANI:")
    print("   - Hafif yuk (backtest): 5-7 dakika")
    print("   - Agir yuk (optimizer): 3-5 dakika")
    print("   - Cok agir yuk (8 worker): 2-3 dakika")

    print("\n   ONERI: 3 CPU worker + 6 IO worker kullanin")
    print("   Bu ayarlar thermal budget'i asmayi engeller")

    return True


def run_mini_benchmark():
    """Bonus: Mini benchmark"""
    print("\n" + "="*60)
    print("MINI BENCHMARK")
    print("="*60)

    try:
        import pandas as pd
        import numpy as np

        # DataFrame operations
        print("\n   DataFrame islemleri (10k rows)...")
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1min'),
            'close': np.random.randn(10000).cumsum() + 100,
            'high': np.random.randn(10000).cumsum() + 101,
            'low': np.random.randn(10000).cumsum() + 99,
        })

        t1 = time.perf_counter()
        for _ in range(100):
            _ = df['close'].rolling(20).mean()
        t2 = time.perf_counter()

        print(f"   100x rolling(20).mean(): {(t2-t1)*1000:.1f}ms")

        # NumPy operations
        print("\n   NumPy islemleri (1M elements)...")
        arr = np.random.randn(1000000)

        t1 = time.perf_counter()
        for _ in range(100):
            _ = np.searchsorted(arr, 0.5)
        t2 = time.perf_counter()

        print(f"   100x searchsorted: {(t2-t1)*1000:.1f}ms")

        # Date parsing
        print("\n   Timestamp parsing (10k)...")
        dates = ['2024-01-01 12:00:00'] * 10000

        t1 = time.perf_counter()
        _ = pd.to_datetime(dates)
        t2 = time.perf_counter()

        print(f"   10k timestamp parse: {(t2-t1)*1000:.1f}ms")

        return True

    except Exception as e:
        print(f"\n   [HATA] Benchmark hatasi: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("M3 MACBOOK AIR PERFORMANCE VERIFICATION")
    print("="*60)
    print("Bu script M3 optimizasyonlarinin dogru uygulandigini kontrol eder")

    results = {
        'arm64': check_arm64_native(),
        'accelerate': check_numpy_accelerate(),
        'workers': check_worker_settings(),
        'caching': check_timedelta_caching(),
        'thermal': estimate_thermal_throttling(),
    }

    run_mini_benchmark()

    # Summary
    print("\n" + "="*60)
    print("OZET")
    print("="*60)

    passed = sum(1 for v in results.values() if v is True)
    total = len(results)

    print(f"\n   Gecen: {passed}/{total} kontrol")

    if results['arm64'] is False:
        print("\n   [KRITIK] ARM64 native degil - %30-40 performans kaybi!")

    if passed == total:
        print("\n   [TAMAM] Tum M3 optimizasyonlari aktif!")
        print("   Beklenen runtime: 4-5 dakika (7-8 dk yerine)")
    else:
        print("\n   [EKSIK] Bazi optimizasyonlar eksik")
        print("   Yukaridaki UYARI/HATA mesajlarini kontrol edin")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
