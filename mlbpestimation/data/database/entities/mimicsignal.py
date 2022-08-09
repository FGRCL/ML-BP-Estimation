from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

from src.data.database.base import Base


class MimicSignal(Base):
    __tablename__ = "mimic_signal"

    id = Column(Integer, primary_key=True)
    record_id = Column(Integer, ForeignKey("mimic_record.id"))

    type = Column(String)

    values = relationship("MimicSignalValue", back_populates="signal", cascade="all, delete-orphan")
    record = relationship("MimicRecord", back_populates="signals")
