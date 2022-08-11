from sqlalchemy import Column, Float, Integer
from sqlalchemy.orm import relationship

from mlbpestimation.data.database.base import Base


class MimicRecord(Base):
    __tablename__ = "mimic_record"

    id = Column(Integer, primary_key=True)

    record_id = Column(Integer)
    frequency = Column(Float)
    length = Column(Integer)

    signals = relationship("MimicSignal", back_populates="record", cascade="all, delete-orphan")
