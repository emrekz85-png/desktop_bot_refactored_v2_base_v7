"""
Experiment Tracking Module for SSL Flow Trading Bot.

Bu modül tüm deneyleri, değişiklikleri ve test sonuçlarını takip eder.
Her test çalıştırıldığında otomatik olarak loglanır ve karşılaştırılabilir.

Kullanım:
    from core.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()

    # Test başlamadan önce
    exp_id = tracker.start_experiment(
        name="ATR RMA Test",
        changes={"ATR_METHOD": "RMA"},
        hypothesis="TradingView uyumu sağlanacak"
    )

    # Test bittikten sonra
    tracker.end_experiment(exp_id, results={
        "pnl": -39.90,
        "trades": 13,
        "win_rate": 30.8,
        "max_dd": 98.06
    })

    # Karşılaştırma
    tracker.compare_with_baseline(exp_id)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib


class ExperimentTracker:
    """Deney takip ve loglama sistemi."""

    # Baseline değerleri (v1.0.0)
    BASELINE = {
        "version": "1.0.0",
        "date": "2025-12-31",
        "pnl": -161.99,
        "trades": 51,
        "win_rate": 41.0,
        "max_dd": 208.0,
        "config": {
            "ATR_METHOD": "SMA",
            "MOMENTUM_SOURCE": "MFI",
            "skip_wick_rejection": False,
            "flat_threshold": 0.001,
            "lookback_days": 60,
        }
    }

    def __init__(self, data_dir: str = None):
        """
        Initialize experiment tracker.

        Args:
            data_dir: Directory to store experiment data.
                      Default: data/experiments/
        """
        if data_dir is None:
            base_dir = Path(__file__).parent.parent
            data_dir = base_dir / "data" / "experiments"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.data_dir / "experiment_history.json"
        self.current_experiment = None

        # Load existing history
        self.history = self._load_history()

    def _load_history(self) -> Dict:
        """Load experiment history from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"experiments": [], "baseline": self.BASELINE}
        return {"experiments": [], "baseline": self.BASELINE}

    def _save_history(self) -> None:
        """Save experiment history to JSON file."""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)

    def _generate_id(self, name: str) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{name}_{timestamp}"
        short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{short_hash}"

    def start_experiment(
        self,
        name: str,
        changes: Dict[str, Any],
        hypothesis: str = "",
        symbols: List[str] = None,
        timeframes: List[str] = None,
        lookback_days: int = 60,
        notes: str = ""
    ) -> str:
        """
        Start a new experiment.

        Args:
            name: Experiment name (e.g., "ATR RMA Test")
            changes: Dict of changes from baseline (e.g., {"ATR_METHOD": "RMA"})
            hypothesis: Expected outcome
            symbols: Symbols being tested
            timeframes: Timeframes being tested
            lookback_days: Lookback period
            notes: Additional notes

        Returns:
            Experiment ID
        """
        exp_id = self._generate_id(name)

        self.current_experiment = {
            "id": exp_id,
            "name": name,
            "status": "running",
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "changes": changes,
            "hypothesis": hypothesis,
            "symbols": symbols or ["BTCUSDT", "ETHUSDT", "LINKUSDT"],
            "timeframes": timeframes or ["15m", "1h"],
            "lookback_days": lookback_days,
            "notes": notes,
            "results": None,
            "comparison": None,
            "verdict": None,
        }

        # Print experiment start banner
        self._print_start_banner()

        return exp_id

    def end_experiment(
        self,
        exp_id: str,
        results: Dict[str, float],
        verdict: str = None,
        notes: str = ""
    ) -> Dict:
        """
        End an experiment and record results.

        Args:
            exp_id: Experiment ID from start_experiment()
            results: Dict with pnl, trades, win_rate, max_dd
            verdict: "success", "failure", "inconclusive"
            notes: Additional notes

        Returns:
            Experiment record with comparison
        """
        if self.current_experiment is None or self.current_experiment["id"] != exp_id:
            raise ValueError(f"No active experiment with ID {exp_id}")

        # Record results
        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = "completed"
        self.current_experiment["results"] = results

        # Compare with baseline
        comparison = self._compare_with_baseline(results)
        self.current_experiment["comparison"] = comparison

        # Auto-determine verdict if not provided
        if verdict is None:
            verdict = self._auto_verdict(comparison)
        self.current_experiment["verdict"] = verdict

        if notes:
            self.current_experiment["notes"] += f"\n{notes}"

        # Save to history
        self.history["experiments"].append(self.current_experiment)
        self._save_history()

        # Print results banner
        self._print_results_banner()

        result = self.current_experiment.copy()
        self.current_experiment = None

        return result

    def _compare_with_baseline(self, results: Dict[str, float]) -> Dict:
        """Compare results with baseline."""
        baseline = self.BASELINE

        pnl_change = results.get("pnl", 0) - baseline["pnl"]
        trades_change = results.get("trades", 0) - baseline["trades"]
        wr_change = results.get("win_rate", 0) - baseline["win_rate"]
        dd_change = results.get("max_dd", 0) - baseline["max_dd"]

        return {
            "pnl_change": pnl_change,
            "pnl_pct_change": (pnl_change / abs(baseline["pnl"])) * 100 if baseline["pnl"] != 0 else 0,
            "trades_change": trades_change,
            "win_rate_change": wr_change,
            "max_dd_change": dd_change,
            "is_pnl_better": pnl_change > 0,
            "is_dd_better": dd_change < 0,
            "is_wr_better": wr_change > 0,
        }

    def _auto_verdict(self, comparison: Dict) -> str:
        """Automatically determine verdict based on comparison."""
        pnl_better = comparison["is_pnl_better"]
        dd_better = comparison["is_dd_better"]

        if pnl_better and dd_better:
            return "success"
        elif pnl_better or dd_better:
            return "partial_success"
        elif comparison["pnl_change"] < -50:  # Significant regression
            return "failure"
        else:
            return "inconclusive"

    def _print_start_banner(self) -> None:
        """Print experiment start banner."""
        exp = self.current_experiment
        print("\n" + "=" * 70)
        print("🧪 DENEY BAŞLADI")
        print("=" * 70)
        print(f"   ID: {exp['id']}")
        print(f"   İsim: {exp['name']}")
        print(f"   Hipotez: {exp['hypothesis']}")
        print(f"   Değişiklikler:")
        for key, value in exp['changes'].items():
            baseline_val = self.BASELINE['config'].get(key, 'N/A')
            print(f"      {key}: {baseline_val} → {value}")
        print(f"   Semboller: {', '.join(exp['symbols'])}")
        print(f"   Timeframes: {', '.join(exp['timeframes'])}")
        print("=" * 70 + "\n")

    def _print_results_banner(self) -> None:
        """Print experiment results banner."""
        exp = self.current_experiment
        results = exp['results']
        comp = exp['comparison']

        verdict_emoji = {
            "success": "✅",
            "partial_success": "🟡",
            "failure": "❌",
            "inconclusive": "❓"
        }

        print("\n" + "=" * 70)
        print(f"🏁 DENEY TAMAMLANDI - {verdict_emoji.get(exp['verdict'], '❓')} {exp['verdict'].upper()}")
        print("=" * 70)
        print(f"   ID: {exp['id']}")
        print(f"   İsim: {exp['name']}")
        print()
        print("   📊 SONUÇLAR:")
        print(f"      PnL: ${results.get('pnl', 0):.2f} ({comp['pnl_change']:+.2f} vs baseline)")
        print(f"      Trades: {results.get('trades', 0)} ({comp['trades_change']:+d} vs baseline)")
        print(f"      Win Rate: {results.get('win_rate', 0):.1f}% ({comp['win_rate_change']:+.1f}% vs baseline)")
        print(f"      Max DD: ${results.get('max_dd', 0):.2f} ({comp['max_dd_change']:+.2f} vs baseline)")
        print()
        print("   📈 DEĞERLENDİRME:")
        print(f"      PnL: {'✅ İyileşti' if comp['is_pnl_better'] else '❌ Kötüleşti'}")
        print(f"      Drawdown: {'✅ Azaldı' if comp['is_dd_better'] else '❌ Arttı'}")
        print(f"      Win Rate: {'✅ Arttı' if comp['is_wr_better'] else '❌ Düştü'}")
        print("=" * 70 + "\n")

    def get_experiment(self, exp_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        for exp in self.history["experiments"]:
            if exp["id"] == exp_id:
                return exp
        return None

    def get_recent_experiments(self, limit: int = 10) -> List[Dict]:
        """Get most recent experiments."""
        return self.history["experiments"][-limit:]

    def get_successful_experiments(self) -> List[Dict]:
        """Get all successful experiments."""
        return [exp for exp in self.history["experiments"]
                if exp.get("verdict") in ["success", "partial_success"]]

    def get_failed_experiments(self) -> List[Dict]:
        """Get all failed experiments."""
        return [exp for exp in self.history["experiments"]
                if exp.get("verdict") == "failure"]

    def print_summary(self) -> None:
        """Print summary of all experiments."""
        experiments = self.history["experiments"]

        if not experiments:
            print("Henüz kaydedilmiş deney yok.")
            return

        total = len(experiments)
        success = len([e for e in experiments if e.get("verdict") == "success"])
        partial = len([e for e in experiments if e.get("verdict") == "partial_success"])
        failed = len([e for e in experiments if e.get("verdict") == "failure"])

        print("\n" + "=" * 70)
        print("📋 DENEY ÖZETİ")
        print("=" * 70)
        print(f"   Toplam Deney: {total}")
        print(f"   ✅ Başarılı: {success}")
        print(f"   🟡 Kısmi Başarı: {partial}")
        print(f"   ❌ Başarısız: {failed}")
        print()

        print("   Son 5 Deney:")
        for exp in experiments[-5:]:
            verdict_emoji = {"success": "✅", "partial_success": "🟡",
                           "failure": "❌", "inconclusive": "❓"}
            emoji = verdict_emoji.get(exp.get("verdict", ""), "❓")
            pnl_change = exp.get("comparison", {}).get("pnl_change", 0)
            print(f"      {emoji} {exp['name']}: {pnl_change:+.2f} PnL")
        print("=" * 70 + "\n")

    def export_to_markdown(self, output_file: str = None) -> str:
        """Export experiment history to markdown."""
        if output_file is None:
            output_file = self.data_dir / "experiment_report.md"

        lines = [
            "# Deney Raporu",
            "",
            f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Baseline",
            "",
            f"- Version: {self.BASELINE['version']}",
            f"- PnL: ${self.BASELINE['pnl']:.2f}",
            f"- Trades: {self.BASELINE['trades']}",
            f"- Win Rate: {self.BASELINE['win_rate']}%",
            f"- Max DD: ${self.BASELINE['max_dd']:.2f}",
            "",
            "## Deneyler",
            "",
        ]

        for exp in self.history["experiments"]:
            verdict_emoji = {"success": "✅", "partial_success": "🟡",
                           "failure": "❌", "inconclusive": "❓"}
            emoji = verdict_emoji.get(exp.get("verdict", ""), "❓")

            lines.extend([
                f"### {emoji} {exp['name']}",
                "",
                f"- ID: `{exp['id']}`",
                f"- Tarih: {exp.get('start_time', 'N/A')[:10]}",
                f"- Hipotez: {exp.get('hypothesis', 'N/A')}",
                "",
                "**Değişiklikler:**",
            ])

            for key, value in exp.get("changes", {}).items():
                lines.append(f"- {key}: {value}")

            results = exp.get("results", {})
            comp = exp.get("comparison", {})

            lines.extend([
                "",
                "**Sonuçlar:**",
                "",
                "| Metrik | Değer | vs Baseline |",
                "|--------|-------|-------------|",
                f"| PnL | ${results.get('pnl', 0):.2f} | {comp.get('pnl_change', 0):+.2f} |",
                f"| Trades | {results.get('trades', 0)} | {comp.get('trades_change', 0):+d} |",
                f"| Win Rate | {results.get('win_rate', 0):.1f}% | {comp.get('win_rate_change', 0):+.1f}% |",
                f"| Max DD | ${results.get('max_dd', 0):.2f} | {comp.get('max_dd_change', 0):+.2f} |",
                "",
                f"**Sonuç:** {exp.get('verdict', 'N/A').upper()}",
                "",
                "---",
                "",
            ])

        content = "\n".join(lines)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(output_file)


# Singleton instance for easy access
_tracker = None

def get_tracker() -> ExperimentTracker:
    """Get singleton ExperimentTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = ExperimentTracker()
    return _tracker


# Convenience functions
def start_experiment(name: str, changes: Dict, **kwargs) -> str:
    """Start a new experiment. See ExperimentTracker.start_experiment()."""
    return get_tracker().start_experiment(name, changes, **kwargs)

def end_experiment(exp_id: str, results: Dict, **kwargs) -> Dict:
    """End an experiment. See ExperimentTracker.end_experiment()."""
    return get_tracker().end_experiment(exp_id, results, **kwargs)

def print_experiment_summary() -> None:
    """Print summary of all experiments."""
    get_tracker().print_summary()
