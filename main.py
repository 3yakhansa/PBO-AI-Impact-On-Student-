from __future__ import annotations

import os
import sys
import subprocess
import time

from repository import StudentRepository
from service import AIStudentAnalysisService

DATASET_PATH = os.path.join(os.path.dirname(__file__), "ai_student_impact_dataset.csv")
CHARTS_DIR   = os.path.join(os.path.dirname(__file__), "charts")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║        AI STUDENT IMPACT ANALYSIS SYSTEM v2.0               ║
║        Object-Oriented Programming — Python                  ║
║        Dataset: AI Student Impact (50,000 students)          ║
╚══════════════════════════════════════════════════════════════╝"""

MENU = """
  ┌─────────────────────────────────────────────────────┐
  │  MENU UTAMA                                         │
  ├─────────────────────────────────────────────────────┤
  │  [1]  Overview Dataset                              │
  │  [2]  Analisis Perubahan GPA           + chart      │
  │  [3]  Analisis Burnout Risk            + chart      │
  │  [4]  Analisis Pola Penggunaan AI      + chart      │
  │  [5]  Analisis Skill Retention         + chart      │
  │  [6]  Analisis Korelasi Variabel       + chart      │
  │  [7]  Jalankan Semua Analisis          + semua chart│
  │  [8]  Export Laporan ke JSON                        │
  │  [9]  Jalankan Unit Tests                           │
  │  [0]  Keluar                                        │
  └─────────────────────────────────────────────────────┘
  Pilih menu: """

SEP = "─" * 64


def _sep():
    print(SEP)


def _open_chart(path: str) -> None:
    """Buka file chart di viewer default sistem operasi."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)
        elif sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _chart_info(path: str, open_chart: bool = False) -> None:
    if path and os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"\n  📊 Chart disimpan: {path}  ({size_kb:.1f} KB)")
        if open_chart:
            _open_chart(path)
    else:
        print(f"\n  ⚠  Chart tidak tersedia.")


# ──────────────────────────────────────────────────────────────
# Display Functions
# ──────────────────────────────────────────────────────────────

def show_overview(service: AIStudentAnalysisService) -> None:
    data = service.get_overview()
    print("\n  ═══ OVERVIEW DATASET ═══")
    print(f"  Total Mahasiswa      : {data['total_students']:,}")
    print(f"  Jurusan              : {', '.join(data['majors'])}")
    print(f"  Tingkat Studi        : {', '.join(data['years'])}")
    print(f"  Kebijakan Institusi  : {', '.join(data['policies'])}")
    print(f"  Rata-rata GPA Awal   : {data['mean_pre_gpa']:.4f}")
    print(f"  Rata-rata GPA Akhir  : {data['mean_post_gpa']:.4f}")
    pct = (data['mean_post_gpa'] - data['mean_pre_gpa']) / data['mean_pre_gpa'] * 100
    print(f"  Perubahan GPA        : {pct:+.2f}%")
    print(f"  Rata-rata Jam AI/mgg : {data['mean_weekly_ai_hours']:.2f} jam")
    print(f"  Pengguna AI Tinggi   : {data['high_ai_users']:,}"
          f" ({data['high_ai_users']/data['total_students']*100:.1f}%)")
    print(f"  Berlangganan Berbayar: {data['paid_subscription']:,}"
          f" ({data['paid_subscription']/data['total_students']*100:.1f}%)")


def show_gpa(service: AIStudentAnalysisService) -> None:
    t0 = time.perf_counter()
    data, chart = service.run_gpa_analysis()
    dur = time.perf_counter() - t0

    print("\n  ═══ ANALISIS PERUBAHAN GPA ═══")
    print(f"  Rata-rata ΔGPA Overall   : {data['mean_gpa_change_overall']:+.4f}")
    print(f"  GPA Awal (mean)          : {data['mean_pre_gpa']:.4f}")
    print(f"  GPA Akhir (mean)         : {data['mean_post_gpa']:.4f}")
    print(f"\n  Pengguna AI Tinggi (>15j): {data['high_ai_users_count']:,}"
          f"  │  ΔGPA = {data['mean_gpa_change_high_ai']:+.4f}")
    print(f"  Pengguna AI Rendah (≤15j): {data['low_ai_users_count']:,}"
          f"  │  ΔGPA = {data['mean_gpa_change_low_ai']:+.4f}")

    print("\n  ΔGPA per Jurusan:")
    for major, chg in data['by_major'].items():
        bar = "█" * max(1, int(abs(chg) * 80))
        sgn = "+" if chg >= 0 else ""
        print(f"    {major:<15}: {sgn}{chg:.4f}  {bar}")

    print("\n  ΔGPA per Tingkat Studi:")
    for year, chg in data['by_year'].items():
        bar = "█" * max(1, int(abs(chg) * 80))
        sgn = "+" if chg >= 0 else ""
        print(f"    {year:<12}: {sgn}{chg:.4f}  {bar}")

    print(f"\n  Selesai dalam {dur:.2f} detik")
    _chart_info(chart, open_chart=True)


def show_burnout(service: AIStudentAnalysisService) -> None:
    t0 = time.perf_counter()
    data, chart = service.run_burnout_analysis()
    dur = time.perf_counter() - t0

    print("\n  ═══ ANALISIS BURNOUT RISK ═══")
    for lvl, info in data['by_level'].items():
        bar = "█" * int(info['rate'] / 2)
        print(f"\n  [{lvl}]  {info['count']:,} mahasiswa ({info['rate']:.1f}%)  {bar}")
        print(f"    ├ AI Hours      : {info['mean_weekly_ai_hours']:.2f} jam/mgg")
        print(f"    ├ Study Hours   : {info['mean_traditional_hours']:.2f} jam/mgg")
        print(f"    ├ Kecemasan     : {info['mean_anxiety']:.2f}/10")
        print(f"    ├ Keterg. AI    : {info['mean_ai_dependency']:.2f}/10")
        print(f"    └ ΔGPA          : {info['mean_gpa_change']:+.4f}")

    print("\n  % Burnout Tinggi per Kebijakan:")
    for pol, info in data['by_policy'].items():
        bar = "█" * int(info['high_burnout_rate'] / 2)
        print(f"    {pol:<30}: {info['high_burnout_rate']:5.1f}%  {bar}")

    print(f"\n  Selesai dalam {dur:.2f} detik")
    _chart_info(chart, open_chart=True)


def show_ai_usage(service: AIStudentAnalysisService) -> None:
    t0 = time.perf_counter()
    data, chart = service.run_ai_usage_analysis()
    dur = time.perf_counter() - t0

    print("\n  ═══ ANALISIS POLA PENGGUNAAN AI ═══")
    print(f"  Rata-rata Jam AI/mgg    : {data['mean_weekly_ai_hours']:.2f}")
    print(f"  Rata-rata Tool Diversity: {data['mean_tool_diversity']:.2f}")
    print(f"  Pengguna Berbayar       : {data['paid_subscription_rate']:.1f}%")
    print(f"  Rata-rata Ketergantungan: {data['mean_ai_dependency']:.2f}/10")
    print(f"  Pengguna AI Tinggi      : {data['high_ai_users_rate']:.1f}%")

    print("\n  ΔGPA & Skill Retention per Use Case:")
    for uc, info in data['by_use_case'].items():
        print(f"    {uc:<30} │ ΔGPA={info['mean_gpa_change']:+.4f} │"
              f" Ret={info['mean_skill_retention']:.1f}%")

    print("\n  Analisis per Prompt Engineering Skill:")
    for lvl, info in data['by_skill_level'].items():
        print(f"    {lvl:<14}: ΔGPA={info['mean_gpa_change']:+.4f} │"
              f" Ret={info['mean_skill_retention']:.1f}% │ AI={info['mean_ai_hours']:.1f}j")

    print(f"\n  Selesai dalam {dur:.2f} detik")
    _chart_info(chart, open_chart=True)


def show_retention(service: AIStudentAnalysisService) -> None:
    t0 = time.perf_counter()
    data, chart = service.run_retention_analysis()
    dur = time.perf_counter() - t0

    print("\n  ═══ ANALISIS SKILL RETENTION ═══")
    print(f"  Rata-rata Retention  : {data['mean_retention']:.2f}%")
    print(f"  Retention Tinggi(≥80%): {data['high_retention_rate']:.1f}%")
    print(f"  Retention Rendah(<60%): {data['low_retention_rate']:.1f}%")

    print("\n  Retention per Prompt Engineering Level:")
    for lvl, score in data['by_skill_level'].items():
        bar = "█" * int(score / 4)
        print(f"    {lvl:<14}: {score:.2f}%  {bar}")

    print("\n  Retention per Jurusan:")
    for major, score in data['by_major'].items():
        bar = "█" * int(score / 4)
        print(f"    {major:<15}: {score:.2f}%  {bar}")

    print(f"\n  Selesai dalam {dur:.2f} detik")
    _chart_info(chart, open_chart=True)


def show_correlation(service: AIStudentAnalysisService) -> None:
    t0 = time.perf_counter()
    data, chart = service.run_correlation_analysis()
    dur = time.perf_counter() - t0

    print("\n  ═══ ANALISIS KORELASI VARIABEL ═══")
    corr_matrix = data["correlation_matrix"]
    target = "gpa_change"

    print(f"\n  Korelasi terhadap '{target}':")
    items = [(k, v[target]) for k, v in corr_matrix.items() if k != target]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    for var, val in items:
        bar_len = int(abs(val) * 20)
        direction = "▶" if val > 0 else "◀"
        print(f"    {var:<25}: {val:+.4f}  {direction}{'█' * bar_len}")

    print(f"\n  Selesai dalam {dur:.2f} detik")
    _chart_info(chart)


# ──────────────────────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────────────────────

def run_tests() -> None:
    from models import GPA, NonNegativeFloat, Student
    from repository import StudentRepository, MajorSpec, HighAIUserSpec
    from analyzers import GPAAnalyzer, BurnoutAnalyzer, AIUsageAnalyzer, RetentionAnalyzer

    print("\n  ═══ UNIT TESTS ═══\n")
    passed = failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            print(f"  ✓  {name}")
            passed += 1
        else:
            print(f"  ✗  GAGAL: {name}")
            failed += 1

    # ── Value Objects ────────────────────────────────────

    check("GPA(3.5) valid", float(GPA(3.5)) == 3.5)
    check("GPA(4.0) valid (boundary)", float(GPA(4.0)) == 4.0)

    try:
        GPA(4.1)
        check("GPA(4.1) harus raise ValueError", False)
    except ValueError:
        check("GPA(4.1) harus raise ValueError", True)

    check("NonNegativeFloat(0) valid", float(NonNegativeFloat(0)) == 0.0)

    try:
        NonNegativeFloat(-1)
        check("NonNegativeFloat(-1) harus raise ValueError", False)
    except ValueError:
        check("NonNegativeFloat(-1) harus raise ValueError", True)

    # ── Student Entity ───────────────────────────────────

    def make_student(**kw) -> Student:
        defaults = dict(
            student_id=1, major_category="STEM", year_of_study="Freshman",
            pre_semester_gpa=GPA(3.0), weekly_genai_hours=NonNegativeFloat(20.0),
            primary_use_case="Ideation", prompt_engineering_skill="Advanced",
            tool_diversity=3, paid_subscription=True,
            traditional_study_hours=NonNegativeFloat(10.0),
            perceived_ai_dependency=5, institutional_policy="Allowed_With_Citation",
            anxiety_level_during_exams=4, post_semester_gpa=GPA(3.5),
            skill_retention_score=85.0, burnout_risk_level="Low"
        )
        defaults.update(kw)
        return Student(**defaults)

    s = make_student()
    check("Student.gpa_change = 0.5", abs(s.gpa_change - 0.5) < 1e-6)
    check("Student.is_high_ai_user (20h)", s.is_high_ai_user)
    check("Student.is_at_burnout_risk = False (Low)", not s.is_at_burnout_risk)
    check("Student.has_high_retention (85%)", s.has_high_retention)
    check("Student.study_intensity rasional", 0 < s.study_intensity <= 1)
    check("Student immutable (frozen)", _is_frozen(s))

    try:
        make_student(burnout_risk_level="Invalid")
        check("burnout tidak valid raise ValueError", False)
    except ValueError:
        check("burnout tidak valid raise ValueError", True)

    # ── Repository & Specification ───────────────────────

    repo = StudentRepository()
    n = repo.load_from_csv(DATASET_PATH)
    check(f"Repository load {n:,} baris", n > 0)

    stem = repo.find_by_major("STEM")
    check("find_by_major('STEM') tidak kosong", len(stem) > 0)

    # Composable spec: STEM AND High AI user
    spec = MajorSpec("STEM") & HighAIUserSpec()
    result = repo.filter_by_spec(spec)
    check("Composed spec (STEM & HighAI) valid", len(result) > 0)
    check("Semua hasil composed spec memenuhi kondisi",
          all(s.major_category == "STEM" and s.is_high_ai_user for s in result))

    # NOT spec
    not_stem = repo.filter_by_spec(~MajorSpec("STEM"))
    check("NOT spec mengecualikan STEM", all(s.major_category != "STEM" for s in not_stem))

    # ── Analyzers ────────────────────────────────────────

    students = repo.get_all()
    for cls, name in [
        (GPAAnalyzer,       "GPAAnalyzer"),
        (BurnoutAnalyzer,   "BurnoutAnalyzer"),
        (AIUsageAnalyzer,   "AIUsageAnalyzer"),
        (RetentionAnalyzer, "RetentionAnalyzer"),
    ]:
        result = cls(students).analyze()
        check(f"{name}.analyze() returns dict", isinstance(result, dict))
        check(f"{name}.analyze() tidak kosong", bool(result))

    # ── Summary ──────────────────────────────────────────

    _sep()
    total_tests = passed + failed
    print(f"\n  Hasil: {passed}/{total_tests} tests passed | {failed} failed")
    if failed == 0:
        print("  ✅  Semua test LULUS!")
    else:
        print(f"  ❌  {failed} test GAGAL.")


def _is_frozen(obj) -> bool:
    """Cek apakah dataclass frozen (assignment harus raise FrozenInstanceError)."""
    try:
        obj.student_id = -999  # type: ignore[misc]
        return False
    except Exception:
        return True


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def _timing_hook(name: str, result: dict, duration: float) -> None:
    """Hook observer sederhana: cetak durasi tiap analisis."""
    print(f"  [Hook] '{name}' selesai dalam {duration:.3f}s")


def main() -> None:
    print(BANNER)

    print(f"\n  Memuat dataset dari: {DATASET_PATH}")
    repo = StudentRepository()
    n    = repo.load_from_csv(DATASET_PATH)
    print(f"  Dataset berhasil dimuat: {n:,} mahasiswa\n")

    service = AIStudentAnalysisService(repo)
    service.add_hook(_timing_hook)   # Observer hook aktif

    print(f"  Service siap: {service}")

    while True:
        print(MENU, end="")
        choice = input().strip()
        _sep()

        dispatch = {
            "1": lambda: show_overview(service),
            "2": lambda: show_gpa(service),
            "3": lambda: show_burnout(service),
            "4": lambda: show_ai_usage(service),
            "5": lambda: show_retention(service),
            "6": lambda: show_correlation(service),
            "7": lambda: _run_all(service),
            "8": lambda: service.export_report("report.json") or print("  Report → report.json"),
            "9": run_tests,
            "0": lambda: _exit(),
        }

        fn = dispatch.get(choice)
        if fn:
            fn()
        else:
            print("  ⚠  Pilihan tidak valid. Coba lagi.")

        _sep()


def _run_all(service: AIStudentAnalysisService) -> None:
    print("\n  Menjalankan semua analisis + membuat semua chart...")
    t0 = time.perf_counter()
    results = service.run_all()
    dur = time.perf_counter() - t0

    show_overview(service)
    print(f"\n  ✅  Semua analisis selesai dalam {dur:.2f} detik")
    if "_charts" in results:
        print(f"\n  Chart yang dihasilkan ({len(results['_charts'])} file):")
        for key, path in results["_charts"].items():
            if path and os.path.exists(path):
                sz = os.path.getsize(path) / 1024
                print(f"    📊 [{key}] {path}  ({sz:.1f} KB)")


def _exit() -> None:
    print("\n  Terima kasih. Program selesai.\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
