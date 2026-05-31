"""
models.py - Model Layer
======================
Mendefinisikan entitas domain dengan enkapsulasi ketat.

Perbaikan OOP:
- Enkapsulasi: field privat via __slots__ + property, setter diblokir (immutable)
- Validasi terpusat di _validate(), bukan tersebar
- Ditambah computed properties yang richer
- __eq__ dan __hash__ untuk kebutuhan set/dict
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar


# ─────────────────────────────────────────────────────────
# Value Objects (immutable primitives dengan validasi)
# ─────────────────────────────────────────────────────────

class GPA(float):
    """
    Value Object untuk GPA.
    Enkapsulasi: validasi rentang di konstruktor, tidak bisa dibuat sembarang.
    """
    MIN: ClassVar[float] = 0.0
    MAX: ClassVar[float] = 4.0

    def __new__(cls, value: float) -> "GPA":
        v = float(value)
        if not (cls.MIN <= v <= cls.MAX):
            raise ValueError(f"GPA harus {cls.MIN}–{cls.MAX}, diberikan: {v}")
        return super().__new__(cls, v)


class NonNegativeFloat(float):
    """Value Object: float yang tidak boleh negatif."""

    def __new__(cls, value: float) -> "NonNegativeFloat":
        v = float(value)
        if v < 0:
            raise ValueError(f"Nilai tidak boleh negatif, diberikan: {v}")
        return super().__new__(cls, v)


# ─────────────────────────────────────────────────────────
# Konstanta Domain
# ─────────────────────────────────────────────────────────

VALID_BURNOUT_LEVELS = frozenset({"Low", "Medium", "High"})
VALID_SKILL_LEVELS   = frozenset({"Beginner", "Intermediate", "Advanced"})
HIGH_AI_THRESHOLD    = 15.0       # jam/minggu
HIGH_RETENTION_THRESHOLD = 80.0   # persen
LOW_RETENTION_THRESHOLD  = 60.0   # persen


# ─────────────────────────────────────────────────────────
# Entity: Student  (frozen=True → immutable setelah dibuat)
# ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Student:
    """
    Entitas mahasiswa — representasi satu baris data.

    OOP Highlights:
    - Enkapsulasi : frozen dataclass → semua field read-only setelah init.
                    Validasi dijalankan di __post_init__ via Value Objects.
    - Abstraksi   : konsumen hanya perlu tahu nama field & properti, bukan
                    detail validasi internal.
    - Computed properties yang kaya (gpa_change, study_intensity, dsb.)
    """

    student_id:               int
    major_category:           str
    year_of_study:            str
    pre_semester_gpa:         GPA
    weekly_genai_hours:       NonNegativeFloat
    primary_use_case:         str
    prompt_engineering_skill: str
    tool_diversity:           int
    paid_subscription:        bool
    traditional_study_hours:  NonNegativeFloat
    perceived_ai_dependency:  int          # 1–10
    institutional_policy:     str
    anxiety_level_during_exams: int        # 1–10
    post_semester_gpa:        GPA
    skill_retention_score:    float        # 0–100
    burnout_risk_level:       str

    # ── post-init validation (Enkapsulasi) ──────────────

    def __post_init__(self) -> None:
        """Validasi cross-field setelah Value Objects divalidasi per-field."""
        errors: list[str] = []

        if self.burnout_risk_level not in VALID_BURNOUT_LEVELS:
            errors.append(f"burnout_risk_level tidak valid: {self.burnout_risk_level!r}")

        if self.prompt_engineering_skill not in VALID_SKILL_LEVELS:
            errors.append(f"prompt_engineering_skill tidak valid: {self.prompt_engineering_skill!r}")

        if not (1 <= self.perceived_ai_dependency <= 10):
            errors.append(f"perceived_ai_dependency harus 1–10: {self.perceived_ai_dependency}")

        if not (1 <= self.anxiety_level_during_exams <= 10):
            errors.append(f"anxiety_level harus 1–10: {self.anxiety_level_during_exams}")

        if not (0.0 <= self.skill_retention_score <= 100.0):
            errors.append(f"skill_retention_score harus 0–100: {self.skill_retention_score}")

        if errors:
            raise ValueError("; ".join(errors))

    # ── Computed Properties (Abstraksi) ─────────────────

    @property
    def gpa_change(self) -> float:
        """Δ GPA (post − pre), dibulatkan 4 desimal."""
        return round(float(self.post_semester_gpa) - float(self.pre_semester_gpa), 4)

    @property
    def is_high_ai_user(self) -> bool:
        """True jika pemakaian AI > threshold mingguan."""
        return float(self.weekly_genai_hours) > HIGH_AI_THRESHOLD

    @property
    def is_at_burnout_risk(self) -> bool:
        """True jika burnout berisiko tinggi."""
        return self.burnout_risk_level == "High"

    @property
    def has_high_retention(self) -> bool:
        """True jika skill retention ≥ threshold tinggi."""
        return self.skill_retention_score >= HIGH_RETENTION_THRESHOLD

    @property
    def has_low_retention(self) -> bool:
        """True jika skill retention < threshold rendah."""
        return self.skill_retention_score < LOW_RETENTION_THRESHOLD

    @property
    def study_intensity(self) -> float:
        """
        Rasio jam AI terhadap total jam belajar (AI + konvensional).
        Mengembalikan 0.0 jika tidak ada jam belajar sama sekali.
        """
        total = float(self.weekly_genai_hours) + float(self.traditional_study_hours)
        return round(float(self.weekly_genai_hours) / total, 4) if total > 0 else 0.0

    @property
    def gpa_category(self) -> str:
        """Kategori GPA akhir: Excellent / Good / Average / Below Average."""
        gpa = float(self.post_semester_gpa)
        if gpa >= 3.7:
            return "Excellent"
        if gpa >= 3.0:
            return "Good"
        if gpa >= 2.0:
            return "Average"
        return "Below Average"

    # ── Dunder Methods ───────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Student(id={self.student_id}, major={self.major_category!r}, "
            f"year={self.year_of_study!r}, gpa_change={self.gpa_change:+.3f}, "
            f"retention={self.skill_retention_score:.1f}%)"
        )