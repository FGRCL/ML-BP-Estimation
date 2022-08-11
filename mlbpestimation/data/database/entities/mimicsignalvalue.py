from sqlalchemy import Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from mlbpestimation.data.database.base import Base


class MimicSignalValue(Base):
    __tablename__ = "mimic_signal_value"

    id = Column(Integer, primary_key=True)
    signal_id = Column(Integer, ForeignKey("mimic_signal.id"))

    value = Column(Float)

    signal = relationship("MimicSignal", back_populates="values")
