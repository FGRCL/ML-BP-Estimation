import sys
from multiprocessing import Pool
from os.path import splitext
from pathlib import Path
from re import match

from numpy import array
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm.auto import tqdm
from wfdb import Record, rdrecord

from src.configuration import Configuration
from src.data.database.base import Base
from src.data.database.entities.mimicrecord import MimicRecord
from src.data.database.entities.mimicsignal import MimicSignal
from src.data.database.entities.mimicsignalvalue import MimicSignalValue


def main():
    engine = create_engine(Configuration.databaseUrl, echo=True)

    Base.metadata.create_all(engine)

    with Session(engine) as session:
        session.query(MimicSignalValue).delete()
        session.query(MimicSignal).delete()
        session.query(MimicRecord).delete()
        session.commit()

    record_paths = get_paths()
    for path in tqdm(record_paths, position=0):
        save_record_entity(path, engine)


def get_paths():
    return [splitext(path)[0] for path in
            Path('../../../data/mimic-IV/physionet.org/files/mimic4wdb/0.1.0/waves').rglob('*hea') if
            match(r'(\d)*.hea', path.name)]


def save_record_entity(path, engine):
    record = rdrecord(path)
    record = map_record_to_entity(record)

    with Session(engine) as session:
        session.add(record)
        session.commit()


def map_record_to_entity(record: Record):
    record_entity = MimicRecord(
        record_id=record.record_name,
        frequency=record.fs,
        length=record.sig_len,
    )

    with Pool() as pool:
        for signal_name, values in tqdm(zip(record.sig_name, array(record.p_signal).T), total=len(record.sig_name),
                                        position=1):
            record_entity.signals.append(map_signal_to_entity(signal_name, values))

    return record_entity


def map_signal_to_entity(signal_name, values):
    signal_entity = MimicSignal(
        type=signal_name
    )

    for value in tqdm(values, position=2):
        value_entity = MimicSignalValue(
            value=value
        )
        signal_entity.values.append(value_entity)

    return signal_entity


if __name__ == "__main__":
    sys.exit(main())