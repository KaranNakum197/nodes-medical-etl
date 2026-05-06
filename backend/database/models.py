"""
SQLAlchemy ORM models for the medical ETL database.
"""

from __future__ import annotations

import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, String, Boolean, Date
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class MedicalRecord(Base):
    """Stores validated medical lab reports."""

    __tablename__ = "medical_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    patient_name: Mapped[str] = mapped_column(String(255), nullable=True)
    report_date: Mapped[datetime.date] = mapped_column(Date, nullable=True)
    test_results: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    anomalies_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "patient_name": self.patient_name,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "test_results": self.test_results,
            "anomalies_flagged": self.anomalies_flagged,
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
        }
