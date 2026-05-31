"""
repository.py - Repository & Specification Layer
=================================================
Memisahkan data-access logic dari business logic (SRP).

Perbaikan OOP & SOLID:
- Repository Pattern: StudentRepository sebagai satu-satunya pintu ke data
- Specification Pattern: kondisi filter dienkapsulasi dalam objek Specification
  → Open-Closed: tambah filter baru cukup tambah Spec baru, repo tidak berubah
- Interface Segregation: IRepository hanya mendefinisikan operasi yang benar-benar
  dibutuhkan consumer
- Dependency Inversion: consumer bergantung pada abstrak IRepository, bukan konkret
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterator, List, TypeVar

from models import GPA, NonNegativeFloat, Student

T = TypeVar("T")


# ─────────────────────────────────────────────────────────
# Abstraksi: Generic Repository Interface  (DIP)
# ─────────────────────────────────────────────────────────

class IRepository(ABC, Generic[T]):
    """
    Interface repository generik.
    Consumer harus bergantung pada abstraksi ini, bukan pada implementasi konkret.
    ISP: hanya method yang benar-benar dipakai consumer yang ada di sini.
    """

    @abstractmethod
    def get_all(self) -> List[T]: ...

    @abstractmethod
    def filter(self, predicate: Callable[[T], bool]) -> List[T]: ...

    @abstractmethod
    def count(self) -> int: ...


# ─────────────────────────────────────────────────────────
# Specification Pattern (OCP)
# ─────────────────────────────────────────────────────────

class Specification(ABC, Generic[T]):
    """
    Abstraksi Specification — enkapsulasi sebuah kondisi filter.
    OCP: menambah filter baru = membuat Spec baru, tidak ubah kode lama.
    """

    @abstractmethod
    def is_satisfied_by(self, item: T) -> bool: ...

    def __call__(self, item: T) -> bool:
        """Agar Spec bisa dipakai langsung sebagai callable predicate."""
        return self.is_satisfied_by(item)

    # Operator komposisi (Polimorfisme via dunder)
    def __and__(self, other: "Specification[T]") -> "AndSpec[T]":
        return AndSpec(self, other)

    def __or__(self, other: "Specification[T]") -> "OrSpec[T]":
        return OrSpec(self, other)

    def __invert__(self) -> "NotSpec[T]":
        return NotSpec(self)


class AndSpec(Specification[T]):
    """Komposisi: kedua spec harus terpenuhi."""
    def __init__(self, a: Specification[T], b: Specification[T]) -> None:
        self._a, self._b = a, b

    def is_satisfied_by(self, item: T) -> bool:
        return self._a.is_satisfied_by(item) and self._b.is_satisfied_by(item)


class OrSpec(Specification[T]):
    """Komposisi: salah satu spec harus terpenuhi."""
    def __init__(self, a: Specification[T], b: Specification[T]) -> None:
        self._a, self._b = a, b

    def is_satisfied_by(self, item: T) -> bool:
        return self._a.is_satisfied_by(item) or self._b.is_satisfied_by(item)


class NotSpec(Specification[T]):
    """Komposisi: spec dinegasi."""
    def __init__(self, spec: Specification[T]) -> None:
        self._spec = spec

    def is_satisfied_by(self, item: T) -> bool:
        return not self._spec.is_satisfied_by(item)


# ─────────────────────────────────────────────────────────
# Concrete Specifications untuk Student
# ─────────────────────────────────────────────────────────

class MajorSpec(Specification[Student]):
    def __init__(self, major: str) -> None:
        self._major = major

    def is_satisfied_by(self, s: Student) -> bool:
        return s.major_category == self._major


class YearSpec(Specification[Student]):
    def __init__(self, year: str) -> None:
        self._year = year

    def is_satisfied_by(self, s: Student) -> bool:
        return s.year_of_study == self._year


class PolicySpec(Specification[Student]):
    def __init__(self, policy: str) -> None:
        self._policy = policy

    def is_satisfied_by(self, s: Student) -> bool:
        return s.institutional_policy == self._policy


class BurnoutSpec(Specification[Student]):
    def __init__(self, level: str) -> None:
        self._level = level

    def is_satisfied_by(self, s: Student) -> bool:
        return s.burnout_risk_level == self._level


class HighAIUserSpec(Specification[Student]):
    def is_satisfied_by(self, s: Student) -> bool:
        return s.is_high_ai_user


class HighRetentionSpec(Specification[Student]):
    def is_satisfied_by(self, s: Student) -> bool:
        return s.has_high_retention


# ─────────────────────────────────────────────────────────
# Concrete Repository
# ─────────────────────────────────────────────────────────

class StudentRepository(IRepository[Student]):
    """
    Repository konkret untuk Student.

    Enkapsulasi : _students bersifat privat; tidak ada akses langsung dari luar.
    SRP         : hanya bertanggung jawab atas penyimpanan & query data.
    OCP         : query baru cukup kirim Specification baru, tidak ubah class ini.
    DIP         : mengimplementasikan IRepository (bukan berdiri sendiri).
    """

    def __init__(self) -> None:
        self.__students: List[Student] = []   # name-mangled → benar-benar privat

    # ── Pemuatan Data ────────────────────────────────────

    def load_from_csv(self, filepath: str) -> int:
        """
        Muat data CSV ke dalam repository.
        Returns jumlah baris yang berhasil dimuat.
        """
        loaded = skipped = 0

        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    student = self._row_to_student(row)
                    self.__students.append(student)
                    loaded += 1
                except (ValueError, KeyError) as exc:
                    skipped += 1

        print(f"  [Repository] Loaded: {loaded:,} | Skipped: {skipped}")
        return loaded

    @staticmethod
    def _row_to_student(row: dict) -> Student:
        """
        Factory method: konversi satu baris CSV → Student.
        Enkapsulasi: logika parsing terisolasi di sini.
        """
        return Student(
            student_id               = int(row["Student_ID"]),
            major_category           = row["Major_Category"].strip(),
            year_of_study            = row["Year_of_Study"].strip(),
            pre_semester_gpa         = GPA(float(row["Pre_Semester_GPA"])),
            weekly_genai_hours       = NonNegativeFloat(float(row["Weekly_GenAI_Hours"])),
            primary_use_case         = row["Primary_Use_Case"].strip(),
            prompt_engineering_skill = row["Prompt_Engineering_Skill"].strip(),
            tool_diversity           = int(row["Tool_Diversity"]),
            paid_subscription        = row["Paid_Subscription"].strip().lower() == "true",
            traditional_study_hours  = NonNegativeFloat(float(row["Traditional_Study_Hours"])),
            perceived_ai_dependency  = int(row["Perceived_AI_Dependency"]),
            institutional_policy     = row["Institutional_Policy"].strip(),
            anxiety_level_during_exams = int(row["Anxiety_Level_During_Exams"]),
            post_semester_gpa        = GPA(float(row["Post_Semester_GPA"])),
            skill_retention_score    = float(row["Skill_Retention_Score"]),
            burnout_risk_level       = row["Burnout_Risk_Level"].strip(),
        )

    # ── IRepository Interface ────────────────────────────

    def get_all(self) -> List[Student]:
        return list(self.__students)       # kembalikan salinan, bukan referensi

    def filter(self, predicate: Callable[[Student], bool]) -> List[Student]:
        return [s for s in self.__students if predicate(s)]

    def filter_by_spec(self, spec: Specification[Student]) -> List[Student]:
        """Query dengan Specification — OCP-friendly."""
        return self.filter(spec)

    def count(self) -> int:
        return len(self.__students)

    # ── Convenience Queries (delegasi ke filter_by_spec) ─

    def find_by_major(self, major: str)   -> List[Student]:
        return self.filter_by_spec(MajorSpec(major))

    def find_by_year(self, year: str)     -> List[Student]:
        return self.filter_by_spec(YearSpec(year))

    def find_by_policy(self, policy: str) -> List[Student]:
        return self.filter_by_spec(PolicySpec(policy))

    def find_by_burnout(self, level: str) -> List[Student]:
        return self.filter_by_spec(BurnoutSpec(level))

    def find_high_ai_users(self)          -> List[Student]:
        return self.filter_by_spec(HighAIUserSpec())

    # ── Iteration Support ────────────────────────────────

    def __iter__(self) -> Iterator[Student]:
        return iter(self.__students)

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"StudentRepository(count={self.count():,})"