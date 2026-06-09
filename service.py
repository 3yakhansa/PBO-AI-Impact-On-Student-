from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from repository import IRepository, StudentRepository
from analyzers import (
    AIUsageAnalyzer,
    BaseStudentAnalyzer,
    BurnoutAnalyzer,
    CorrelationAnalyzer,
    GPAAnalyzer,
    RetentionAnalyzer,
)


# ─────────────────────────────────────────────────────────
# Observer Hook Type
# ─────────────────────────────────────────────────────────

AnalysisHook = Callable[[str, Dict[str, Any], float], None]
"""
Tipe callback: (nama_analyzer, hasil_dict, durasi_detik) → None
Observer Pattern: siapapun bisa subscribe ke event selesai analisis.
"""


# ─────────────────────────────────────────────────────────
# Facade: AIStudentAnalysisService
# ─────────────────────────────────────────────────────────

class AIStudentAnalysisService:
    """
    Service layer — Facade Pattern.

    Menyederhanakan interface ke sistem analyzer yang kompleks.
    Consumer cukup panggil run_gpa_analysis(), run_burnout_analysis(), dsb.
    tanpa harus tahu cara kerja internal setiap analyzer.

    DIP  : menerima IRepository, bukan StudentRepository konkret.
    OCP  : tambah analyzer baru = tambah entry di _build_registry(),
           tidak ubah metode lain.
    SRP  : service hanya orkestrator; tidak ada logika kalkulasi di sini.
    Observer : listener bisa didaftarkan via add_hook().
    """

    def __init__(self, repository: IRepository) -> None:
        self._repo      = repository
        self._hooks: List[AnalysisHook] = []

        students = self._repo.get_all()
        # Registry analyzer: (key, instance) — OCP-friendly
        self._registry: Dict[str, BaseStudentAnalyzer] = self._build_registry(students)

    # ── Builder Registry (OCP) ───────────────────────────

    @staticmethod
    def _build_registry(students) -> Dict[str, BaseStudentAnalyzer]:
        """
        Satu tempat mendaftarkan analyzer.
        Tambah analyzer baru = tambah satu baris di sini saja.
        """
        return {
            "gpa":         GPAAnalyzer(students),
            "burnout":     BurnoutAnalyzer(students),
            "ai_usage":    AIUsageAnalyzer(students),
            "retention":   RetentionAnalyzer(students),
            "correlation": CorrelationAnalyzer(students),
        }

    # ── Observer: hook management ────────────────────────

    def add_hook(self, fn: AnalysisHook) -> None:
        """Daftarkan listener yang akan dipanggil setiap analisis selesai."""
        self._hooks.append(fn)

    def _notify(self, name: str, result: Dict[str, Any], duration: float) -> None:
        for fn in self._hooks:
            try:
                fn(name, result, duration)
            except Exception:
                pass  

    # ── Facade Methods ───────────────────────────────────

    def get_overview(self) -> Dict[str, Any]:
        """Statistik ringkasan dataset — tidak memerlukan analyzer."""
        students = self._repo.get_all()
        total    = len(students)
        return {
            "total_students":      total,
            "majors":              sorted(set(s.major_category    for s in students)),
            "years":               sorted(set(s.year_of_study     for s in students)),
            "policies":            sorted(set(s.institutional_policy for s in students)),
            "mean_pre_gpa":        round(sum(float(s.pre_semester_gpa)  for s in students) / total, 4),
            "mean_post_gpa":       round(sum(float(s.post_semester_gpa) for s in students) / total, 4),
            "mean_weekly_ai_hours": round(sum(float(s.weekly_genai_hours) for s in students) / total, 2),
            "high_ai_users":       sum(1 for s in students if s.is_high_ai_user),
            "paid_subscription":   sum(1 for s in students if s.paid_subscription),
        }

    def _run_analyzer(self, key: str) -> Tuple[Dict[str, Any], str]:
        """
        Jalankan analyzer berdasarkan key, ukur durasi, notifikasi hook.
        Returns (result_dict, chart_path).
        """
        analyzer = self._registry[key]
        t0       = time.perf_counter()
        result, chart_path = analyzer.run()
        duration = time.perf_counter() - t0
        self._notify(key, result, duration)
        return result, chart_path

    def run_gpa_analysis(self)         -> Tuple[Dict[str, Any], str]:
        return self._run_analyzer("gpa")

    def run_burnout_analysis(self)     -> Tuple[Dict[str, Any], str]:
        return self._run_analyzer("burnout")

    def run_ai_usage_analysis(self)    -> Tuple[Dict[str, Any], str]:
        return self._run_analyzer("ai_usage")

    def run_retention_analysis(self)   -> Tuple[Dict[str, Any], str]:
        return self._run_analyzer("retention")

    def run_correlation_analysis(self) -> Tuple[Dict[str, Any], str]:
        return self._run_analyzer("correlation")

    def run_all(self) -> Dict[str, Any]:
        """Jalankan seluruh registry analyzer sekaligus; return dict hasil."""
        results = {"overview": self.get_overview()}
        charts  = {}
        for key in self._registry:
            result, chart_path = self._run_analyzer(key)
            results[key] = result
            charts[key]  = chart_path
        results["_charts"] = charts
        return results

    # ── Export ───────────────────────────────────────────

    def export_report(self, filepath: str = "report.json") -> None:
        """Ekspor semua hasil analisis ke JSON (tanpa path chart internal)."""
        all_results = self.run_all()
        # Hapus key internal sebelum ekspor
        exportable = {k: v for k, v in all_results.items() if not k.startswith("_")}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(exportable, f, indent=2, ensure_ascii=False)
        print(f"  [Service] Report diekspor ke: {filepath}")

    # ── Dunder ───────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AIStudentAnalysisService("
            f"students={self._repo.count():,}, "
            f"analyzers={list(self._registry.keys())})"
        )
