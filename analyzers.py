from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

from models import Student

# ── Konfigurasi Visual Global ────────────────────────────────────────────────

PALETTE = {
    "primary":    "#4F81BD",
    "secondary":  "#C0504D",
    "tertiary":   "#9BBB59",
    "accent":     "#F79646",
    "neutral":    "#8064A2",
    "background": "#F5F5F5",
    "text":       "#2C2C2C",
}
BURNOUT_COLORS = {"High": "#C0504D", "Medium": "#F79646", "Low": "#9BBB59"}
SKILL_COLORS   = {"Beginner": "#8064A2", "Intermediate": "#4F81BD", "Advanced": "#9BBB59"}
YEAR_ORDER     = ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor":  PALETTE["background"],
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "DejaVu Sans",
    "figure.dpi":        120,
})

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save_fig(name: str) -> str:
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    plt.savefig(path, bbox_inches="tight", facecolor=PALETTE["background"])
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base Class  (Abstraksi + Template Method)
# ─────────────────────────────────────────────────────────────────────────────

class BaseStudentAnalyzer(ABC):
    """
    Kelas abstrak untuk semua analyzer mahasiswa.

    Template Method Pattern:
        run() = analyze() + visualize()  → alur baku dari base class.
        Subclass hanya override analyze() dan visualize().

    Helper statistik (_mean, _count_where, _rate, _group_means) diwarisi
    semua subclass → Pewarisan menghindari duplikasi kode.
    """

    def __init__(self, students: List[Student]) -> None:
        if not students:
            raise ValueError("Data mahasiswa tidak boleh kosong.")
        self._students: List[Student] = students   # protected, bukan publik

    # ── Template Method (alur baku) ──────────────────────

    def run(self) -> Tuple[Dict[str, Any], str]:
        """
        Template Method: jalankan analisis + buat visualisasi.
        Returns (hasil_dict, path_chart).
        """
        result = self.analyze()
        chart_path = self.visualize()
        return result, chart_path

    # ── Abstract Methods (wajib diimplementasikan subclass) ─

    @abstractmethod
    def analyze(self) -> Dict[str, Any]:
        """Kalkulasi statistik domain — wajib di-override."""
        ...

    @abstractmethod
    def visualize(self) -> str:
        """Buat chart matplotlib dan simpan; return path file."""
        ...

    @abstractmethod
    def title(self) -> str:
        """Judul singkat untuk laporan/heading."""
        ...

    # ── Protected Helpers (Pewarisan) ────────────────────

    def _mean(self, values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def _count_where(self, pred) -> int:
        return sum(1 for s in self._students if pred(s))

    def _rate(self, count: int, total: int) -> float:
        return round(count / total * 100, 2) if total else 0.0

    def _group_means(self, group_key, value_key) -> Dict[str, float]:
        """
        Helper generik: kelompokkan _students berdasarkan group_key (callable),
        lalu hitung mean value_key (callable) per kelompok.
        """
        groups: Dict[str, List[float]] = {}
        for s in self._students:
            k = group_key(s)
            groups.setdefault(k, []).append(value_key(s))
        return {k: self._mean(v) for k, v in sorted(groups.items())}

    def _sorted_by_order(self, d: Dict[str, Any], order: List[str]) -> Dict[str, Any]:
        """Urutkan dict berdasarkan list order yang diberikan."""
        return {k: d[k] for k in order if k in d}


# ─────────────────────────────────────────────────────────────────────────────
# Subclass 1: GPAAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class GPAAnalyzer(BaseStudentAnalyzer):
    """
    Analisis perubahan GPA berdasarkan intensitas penggunaan AI.
    Polimorfisme: analyze() & visualize() punya logika khusus GPA.
    """

    def title(self) -> str:
        return "Analisis Perubahan GPA"

    def analyze(self) -> Dict[str, Any]:
        all_changes = [s.gpa_change for s in self._students]
        high_ai = [s for s in self._students if s.is_high_ai_user]
        low_ai  = [s for s in self._students if not s.is_high_ai_user]

        return {
            "mean_gpa_change_overall":  self._mean(all_changes),
            "high_ai_users_count":      len(high_ai),
            "low_ai_users_count":       len(low_ai),
            "mean_gpa_change_high_ai":  self._mean([s.gpa_change for s in high_ai]),
            "mean_gpa_change_low_ai":   self._mean([s.gpa_change for s in low_ai]),
            "mean_pre_gpa":             self._mean([float(s.pre_semester_gpa)  for s in self._students]),
            "mean_post_gpa":            self._mean([float(s.post_semester_gpa) for s in self._students]),
            "by_major":                 self._group_means(lambda s: s.major_category, lambda s: s.gpa_change),
            "by_year":                  self._sorted_by_order(
                                            self._group_means(lambda s: s.year_of_study, lambda s: s.gpa_change),
                                            YEAR_ORDER
                                        ),
            "gpa_category_distribution": self._gpa_category_dist(),
        }

    def _gpa_category_dist(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for s in self._students:
            dist[s.gpa_category] = dist.get(s.gpa_category, 0) + 1
        return dict(sorted(dist.items()))

    def visualize(self) -> str:
        data = self.analyze()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("📊 Analisis Perubahan GPA", fontsize=16, fontweight="bold",
                     color=PALETTE["text"], y=1.02)

        # 1. Bar: ΔGPA per Major
        ax = axes[0, 0]
        majors = list(data["by_major"].keys())
        changes = list(data["by_major"].values())
        colors = [PALETTE["tertiary"] if v >= 0 else PALETTE["secondary"] for v in changes]
        bars = ax.barh(majors, changes, color=colors, edgecolor="white", linewidth=0.8)
        ax.axvline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.5)
        ax.bar_label(bars, fmt="{:+.4f}", padding=3, fontsize=8)
        ax.set_title("Rata-rata ΔGPA per Jurusan")
        ax.set_xlabel("ΔGPA")

        # 2. Bar: ΔGPA per Tahun
        ax = axes[0, 1]
        years   = [y for y in YEAR_ORDER if y in data["by_year"]]
        changes2 = [data["by_year"][y] for y in years]
        colors2 = [PALETTE["tertiary"] if v >= 0 else PALETTE["secondary"] for v in changes2]
        bars2 = ax.bar(years, changes2, color=colors2, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.5)
        ax.bar_label(bars2, fmt="{:+.4f}", padding=3, fontsize=8)
        ax.set_title("Rata-rata ΔGPA per Tingkat Studi")
        ax.set_ylabel("ΔGPA")
        ax.tick_params(axis="x", rotation=15)

        # 3. Bar: High vs Low AI users ΔGPA
        ax = axes[1, 0]
        labels  = [f"High AI\n(>15 jam)\nn={data['high_ai_users_count']:,}",
                   f"Low AI\n(≤15 jam)\nn={data['low_ai_users_count']:,}"]
        vals    = [data["mean_gpa_change_high_ai"], data["mean_gpa_change_low_ai"]]
        clrs    = [PALETTE["primary"], PALETTE["accent"]]
        b = ax.bar(labels, vals, color=clrs, edgecolor="white", width=0.5)
        ax.axhline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.5)
        ax.bar_label(b, fmt="{:+.4f}", padding=3, fontsize=9)
        ax.set_title("ΔGPA: Pengguna AI Tinggi vs Rendah")
        ax.set_ylabel("ΔGPA")

        # 4. Pie: Distribusi kategori GPA akhir
        ax = axes[1, 1]
        cat_order = ["Excellent", "Good", "Average", "Below Average"]
        cat_colors = [PALETTE["tertiary"], PALETTE["primary"], PALETTE["accent"], PALETTE["secondary"]]
        cat_data = data["gpa_category_distribution"]
        labels4 = [k for k in cat_order if k in cat_data]
        sizes4  = [cat_data[k] for k in labels4]
        clrs4   = [cat_colors[cat_order.index(k)] for k in labels4]
        wedges, texts, autotexts = ax.pie(
            sizes4, labels=labels4, colors=clrs4, autopct="%1.1f%%",
            startangle=140, pctdistance=0.82, wedgeprops={"edgecolor": "white", "linewidth": 1.5}
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Distribusi Kategori GPA Akhir")

        plt.tight_layout()
        return _save_fig("gpa_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Subclass 2: BurnoutAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class BurnoutAnalyzer(BaseStudentAnalyzer):
    """
    Analisis burnout risk, kecemasan, dan ketergantungan AI.
    Polimorfisme: analyze() & visualize() dengan logika domain burnout.
    """

    def title(self) -> str:
        return "Analisis Burnout Risk"

    def analyze(self) -> Dict[str, Any]:
        total  = len(self._students)
        levels = ["High", "Medium", "Low"]

        by_level: Dict[str, Any] = {}
        for lvl in levels:
            group = [s for s in self._students if s.burnout_risk_level == lvl]
            by_level[lvl] = {
                "count":                  len(group),
                "rate":                   self._rate(len(group), total),
                "mean_weekly_ai_hours":   self._mean([float(s.weekly_genai_hours) for s in group]),
                "mean_traditional_hours": self._mean([float(s.traditional_study_hours) for s in group]),
                "mean_anxiety":           self._mean([s.anxiety_level_during_exams for s in group]),
                "mean_ai_dependency":     self._mean([s.perceived_ai_dependency for s in group]),
                "mean_gpa_change":        self._mean([s.gpa_change for s in group]),
            }

        return {
            "total_students": total,
            "by_level":       by_level,
            "by_policy":      self._burnout_by_policy(),
            "by_major":       self._burnout_by_major(),
        }

    def _burnout_by_policy(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for pol in sorted(set(s.institutional_policy for s in self._students)):
            group = [s for s in self._students if s.institutional_policy == pol]
            high  = [s for s in group if s.is_at_burnout_risk]
            result[pol] = {
                "total":            len(group),
                "high_burnout_rate": self._rate(len(high), len(group)),
            }
        return result

    def _burnout_by_major(self) -> Dict[str, float]:
        return {
            m: self._rate(
                sum(1 for s in self._students if s.major_category == m and s.is_at_burnout_risk),
                sum(1 for s in self._students if s.major_category == m)
            )
            for m in sorted(set(s.major_category for s in self._students))
        }

    def visualize(self) -> str:
        data = self.analyze()
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("🔥 Analisis Burnout Risk", fontsize=16, fontweight="bold",
                     color=PALETTE["text"], y=1.02)

        levels = ["High", "Medium", "Low"]

        # 1. Donut: distribusi burnout level
        ax = axes[0, 0]
        sizes  = [data["by_level"][l]["count"] for l in levels]
        clrs   = [BURNOUT_COLORS[l] for l in levels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=levels, colors=clrs, autopct="%1.1f%%",
            startangle=90, wedgeprops={"width": 0.55, "edgecolor": "white"},
            pctdistance=0.75
        )
        for at in autotexts:
            at.set_fontsize(9)
        ax.set_title("Distribusi Burnout Level")

        # 2. Grouped bar: AI hours & traditional hours per level
        ax = axes[0, 1]
        x    = np.arange(len(levels))
        w    = 0.35
        ai_h = [data["by_level"][l]["mean_weekly_ai_hours"]   for l in levels]
        tr_h = [data["by_level"][l]["mean_traditional_hours"] for l in levels]
        b1 = ax.bar(x - w/2, ai_h, w, label="AI Hours",         color=PALETTE["primary"],  edgecolor="white")
        b2 = ax.bar(x + w/2, tr_h, w, label="Traditional Hours", color=PALETTE["accent"],   edgecolor="white")
        ax.bar_label(b1, fmt="%.1f", padding=2, fontsize=8)
        ax.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(levels)
        ax.set_title("Jam Belajar per Burnout Level")
        ax.set_ylabel("Jam/Minggu")
        ax.legend(fontsize=8)

        # 3. Bar: anxiety & AI dependency per level
        ax = axes[1, 0]
        anxiety = [data["by_level"][l]["mean_anxiety"]       for l in levels]
        depend  = [data["by_level"][l]["mean_ai_dependency"] for l in levels]
        b3 = ax.bar(x - w/2, anxiety, w, label="Kecemasan Ujian", color=PALETTE["secondary"], edgecolor="white")
        b4 = ax.bar(x + w/2, depend,  w, label="Keterg. AI",      color=PALETTE["neutral"],   edgecolor="white")
        ax.bar_label(b3, fmt="%.2f", padding=2, fontsize=8)
        ax.bar_label(b4, fmt="%.2f", padding=2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(levels)
        ax.set_title("Kecemasan & Ketergantungan AI per Burnout Level")
        ax.set_ylabel("Skor (1–10)")
        ax.legend(fontsize=8)

        # 4. Horizontal bar: % burnout tinggi per major
        ax = axes[1, 1]
        majors = list(data["by_major"].keys())
        rates  = list(data["by_major"].values())
        clrs2  = [BURNOUT_COLORS["High"]] * len(majors)
        bars   = ax.barh(majors, rates, color=clrs2, edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        ax.set_title("% Burnout Tinggi per Jurusan")
        ax.set_xlabel("% High Burnout")

        plt.tight_layout()
        return _save_fig("burnout_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Subclass 3: AIUsageAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class AIUsageAnalyzer(BaseStudentAnalyzer):
    """
    Analisis pola penggunaan AI: use case, skill level, kebijakan institusi.
    Polimorfisme: analyze() & visualize() dengan konteks AI-usage.
    """

    def title(self) -> str:
        return "Analisis Pola Penggunaan AI"

    def analyze(self) -> Dict[str, Any]:
        total = len(self._students)
        paid  = [s for s in self._students if s.paid_subscription]

        return {
            "mean_weekly_ai_hours":  self._mean([float(s.weekly_genai_hours) for s in self._students]),
            "mean_tool_diversity":   self._mean([s.tool_diversity for s in self._students]),
            "paid_subscription_rate": self._rate(len(paid), total),
            "mean_ai_dependency":    self._mean([s.perceived_ai_dependency for s in self._students]),
            "high_ai_users_rate":    self._rate(self._count_where(lambda s: s.is_high_ai_user), total),
            "by_use_case":           self._by_use_case(),
            "by_skill_level":        self._by_skill_level(),
            "by_policy":             self._by_policy(),
        }

    def _by_use_case(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for uc in sorted(set(s.primary_use_case for s in self._students)):
            group = [s for s in self._students if s.primary_use_case == uc]
            result[uc] = {
                "count":               len(group),
                "mean_ai_hours":       self._mean([float(s.weekly_genai_hours) for s in group]),
                "mean_gpa_change":     self._mean([s.gpa_change for s in group]),
                "mean_skill_retention": self._mean([s.skill_retention_score for s in group]),
            }
        return result

    def _by_skill_level(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for lvl in ["Beginner", "Intermediate", "Advanced"]:
            group = [s for s in self._students if s.prompt_engineering_skill == lvl]
            result[lvl] = {
                "count":               len(group),
                "mean_gpa_change":     self._mean([s.gpa_change for s in group]),
                "mean_skill_retention": self._mean([s.skill_retention_score for s in group]),
                "mean_ai_hours":       self._mean([float(s.weekly_genai_hours) for s in group]),
            }
        return result

    def _by_policy(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        for pol in sorted(set(s.institutional_policy for s in self._students)):
            group = [s for s in self._students if s.institutional_policy == pol]
            result[pol] = {
                "count":           len(group),
                "mean_ai_hours":   self._mean([float(s.weekly_genai_hours) for s in group]),
                "mean_gpa_change": self._mean([s.gpa_change for s in group]),
            }
        return result

    def visualize(self) -> str:
        data = self.analyze()
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("🤖 Analisis Pola Penggunaan AI", fontsize=16, fontweight="bold",
                     color=PALETTE["text"], y=1.02)

        # 1. Horizontal bar: AI hours per use case
        ax = axes[0, 0]
        uc_data = data["by_use_case"]
        uc_names = list(uc_data.keys())
        uc_hours = [uc_data[u]["mean_ai_hours"] for u in uc_names]
        bars = ax.barh(uc_names, uc_hours, color=PALETTE["primary"], edgecolor="white")
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title("Rata-rata Jam AI per Use Case")
        ax.set_xlabel("Jam/Minggu")

        # 2. Scatter: ΔGPA vs Skill Retention per use case (bubble)
        ax = axes[0, 1]
        for uc, info in uc_data.items():
            ax.scatter(info["mean_gpa_change"], info["mean_skill_retention"],
                       s=info["count"] / 30, alpha=0.75, label=uc,
                       edgecolors="white", linewidths=0.8)
        ax.axvline(0, color=PALETTE["text"], linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title("ΔGPA vs Skill Retention per Use Case\n(ukuran = jumlah mahasiswa)")
        ax.set_xlabel("ΔGPA")
        ax.set_ylabel("Skill Retention (%)")
        ax.legend(fontsize=7, loc="lower right")

        # 3. Grouped bar: ΔGPA & Retention per Skill Level
        ax = axes[1, 0]
        sl_data = data["by_skill_level"]
        lvls    = ["Beginner", "Intermediate", "Advanced"]
        x       = np.arange(len(lvls))
        w       = 0.35
        gpa_ch  = [sl_data[l]["mean_gpa_change"]      for l in lvls]
        ret     = [sl_data[l]["mean_skill_retention"] / 100 for l in lvls]  # scale untuk readability
        b1 = ax.bar(x - w/2, gpa_ch, w, label="ΔGPA",
                    color=[SKILL_COLORS[l] for l in lvls], edgecolor="white")
        ax2_twin = ax.twinx()
        ax2_twin.plot(x, [sl_data[l]["mean_skill_retention"] for l in lvls],
                      "D--", color=PALETTE["secondary"], label="Retention (%)", linewidth=2, markersize=8)
        ax.bar_label(b1, fmt="{:+.4f}", padding=2, fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(lvls)
        ax.set_title("ΔGPA & Skill Retention per Prompt Engineering Skill")
        ax.set_ylabel("ΔGPA"); ax2_twin.set_ylabel("Retention (%)")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        # 4. Bar: Jam AI per kebijakan institusi dengan overlay ΔGPA
        ax = axes[1, 1]
        pol_data = data["by_policy"]
        policies = list(pol_data.keys())
        pol_hours = [pol_data[p]["mean_ai_hours"]   for p in policies]
        pol_gpa   = [pol_data[p]["mean_gpa_change"] for p in policies]
        short_pol = [p.replace("_", "\n") for p in policies]
        bars4 = ax.bar(short_pol, pol_hours, color=PALETTE["neutral"], edgecolor="white")
        ax.bar_label(bars4, fmt="%.1fj", padding=2, fontsize=8)
        ax3_twin = ax.twinx()
        ax3_twin.plot(short_pol, pol_gpa, "o-", color=PALETTE["secondary"],
                      linewidth=2, markersize=8, label="ΔGPA")
        ax.set_title("Jam AI & ΔGPA per Kebijakan Institusi")
        ax.set_ylabel("Jam AI/Minggu"); ax3_twin.set_ylabel("ΔGPA")
        ax.tick_params(axis="x", labelsize=8)
        ax3_twin.legend(fontsize=8)

        plt.tight_layout()
        return _save_fig("ai_usage_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Subclass 4: RetentionAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class RetentionAnalyzer(BaseStudentAnalyzer):
    """
    Analisis skill retention score mahasiswa.
    Polimorfisme: analyze() & visualize() dengan konteks skill retention.
    """

    def title(self) -> str:
        return "Analisis Skill Retention"

    def analyze(self) -> Dict[str, Any]:
        scores = [s.skill_retention_score for s in self._students]
        return {
            "mean_retention":      self._mean(scores),
            "high_retention_rate": self._rate(sum(1 for sc in scores if sc >= 80), len(scores)),
            "low_retention_rate":  self._rate(sum(1 for sc in scores if sc < 60),  len(scores)),
            "by_skill_level":      self._retention_by_skill(),
            "by_major":            self._group_means(lambda s: s.major_category, lambda s: s.skill_retention_score),
            "by_use_case":         self._group_means(lambda s: s.primary_use_case, lambda s: s.skill_retention_score),
            "by_burnout":          self._group_means(lambda s: s.burnout_risk_level, lambda s: s.skill_retention_score),
            "distribution_bins":   self._distribution_bins(scores),
        }

    def _retention_by_skill(self) -> Dict[str, float]:
        return {
            lvl: self._mean([s.skill_retention_score for s in self._students
                             if s.prompt_engineering_skill == lvl])
            for lvl in ["Beginner", "Intermediate", "Advanced"]
        }

    @staticmethod
    def _distribution_bins(scores: List[float]) -> Dict[str, int]:
        bins = {"<40": 0, "40–59": 0, "60–79": 0, "80–89": 0, "≥90": 0}
        for sc in scores:
            if sc < 40:       bins["<40"]   += 1
            elif sc < 60:     bins["40–59"] += 1
            elif sc < 80:     bins["60–79"] += 1
            elif sc < 90:     bins["80–89"] += 1
            else:             bins["≥90"]   += 1
        return bins

    def visualize(self) -> str:
        data  = self.analyze()
        scores = [s.skill_retention_score for s in self._students]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("📚 Analisis Skill Retention", fontsize=16, fontweight="bold",
                     color=PALETTE["text"], y=1.02)

        # 1. Histogram distribusi retention
        ax = axes[0, 0]
        ax.hist(scores, bins=30, color=PALETTE["primary"], edgecolor="white",
                alpha=0.85, linewidth=0.8)
        ax.axvline(data["mean_retention"], color=PALETTE["secondary"], linewidth=2,
                   linestyle="--", label=f'Mean={data["mean_retention"]:.1f}%')
        ax.axvline(80, color=PALETTE["tertiary"], linewidth=1.5, linestyle=":",
                   label="Threshold Tinggi (80%)")
        ax.axvline(60, color=PALETTE["accent"], linewidth=1.5, linestyle=":",
                   label="Threshold Rendah (60%)")
        ax.set_title("Distribusi Skill Retention Score")
        ax.set_xlabel("Retention Score (%)")
        ax.set_ylabel("Jumlah Mahasiswa")
        ax.legend(fontsize=8)

        # 2. Bar: retention per skill level
        ax = axes[0, 1]
        lvls = ["Beginner", "Intermediate", "Advanced"]
        vals = [data["by_skill_level"][l] for l in lvls]
        clrs = [SKILL_COLORS[l] for l in lvls]
        bars = ax.bar(lvls, vals, color=clrs, edgecolor="white", width=0.5)
        ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=9)
        ax.set_ylim(min(vals) * 0.97, max(vals) * 1.02)
        ax.set_title("Skill Retention per Prompt Engineering Level")
        ax.set_ylabel("Mean Retention (%)")

        # 3. Horizontal bar: retention per major
        ax = axes[1, 0]
        majors = list(data["by_major"].keys())
        ret_m  = list(data["by_major"].values())
        colors = sns.color_palette("Blues_d", len(majors))
        bars2  = ax.barh(majors, ret_m, color=colors, edgecolor="white")
        ax.bar_label(bars2, fmt="%.2f%%", padding=3, fontsize=8)
        ax.set_title("Skill Retention per Jurusan")
        ax.set_xlabel("Mean Retention (%)")

        # 4. Bar: retention per burnout level
        ax = axes[1, 1]
        burn_lvls = ["High", "Medium", "Low"]
        burn_vals = [data["by_burnout"].get(l, 0) for l in burn_lvls]
        clrs3 = [BURNOUT_COLORS[l] for l in burn_lvls]
        bars3 = ax.bar(burn_lvls, burn_vals, color=clrs3, edgecolor="white", width=0.5)
        ax.bar_label(bars3, fmt="%.2f%%", padding=3, fontsize=9)
        ax.set_title("Skill Retention per Burnout Level")
        ax.set_ylabel("Mean Retention (%)")
        ax.set_ylim(min(burn_vals) * 0.97, max(burn_vals) * 1.02)

        plt.tight_layout()
        return _save_fig("retention_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# Bonus: CorrelationAnalyzer (demonstrasi OCP — tambah analyzer baru)
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationAnalyzer(BaseStudentAnalyzer):
    """
    Heatmap korelasi antar variabel numerik.
    OCP: class baru tanpa mengubah kode analyzer lain sama sekali.
    """

    def title(self) -> str:
        return "Analisis Korelasi Variabel"

    def analyze(self) -> Dict[str, Any]:
        fields = {
            "weekly_ai_hours":       lambda s: float(s.weekly_genai_hours),
            "traditional_hours":     lambda s: float(s.traditional_study_hours),
            "gpa_change":            lambda s: s.gpa_change,
            "skill_retention":       lambda s: s.skill_retention_score,
            "anxiety":               lambda s: s.anxiety_level_during_exams,
            "ai_dependency":         lambda s: s.perceived_ai_dependency,
            "tool_diversity":        lambda s: s.tool_diversity,
            "study_intensity":       lambda s: s.study_intensity,
        }
        matrix_data = {k: [fn(s) for s in self._students] for k, fn in fields.items()}

        # Hitung korelasi Pearson sederhana tanpa numpy/pandas
        keys = list(matrix_data.keys())
        n    = len(self._students)

        def pearson(x, y):
            mx, my = sum(x) / n, sum(y) / n
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
            dx  = (sum((xi - mx)**2 for xi in x)) ** 0.5
            dy  = (sum((yi - my)**2 for yi in y)) ** 0.5
            return round(num / (dx * dy), 4) if dx * dy else 0.0

        corr: Dict[str, Dict[str, float]] = {}
        for k1 in keys:
            corr[k1] = {k2: pearson(matrix_data[k1], matrix_data[k2]) for k2 in keys}

        return {"correlation_matrix": corr, "variables": keys}

    def visualize(self) -> str:
        data   = self.analyze()
        keys   = data["variables"]
        matrix = np.array([[data["correlation_matrix"][r][c] for c in keys] for r in keys])

        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="RdYlGn",
            xticklabels=[k.replace("_", "\n") for k in keys],
            yticklabels=[k.replace("_", "\n") for k in keys],
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, linecolor="#DDDDDD",
            annot_kws={"size": 8}
        )
        ax.set_title("Heatmap Korelasi Antar Variabel Numerik", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return _save_fig("correlation_heatmap")
