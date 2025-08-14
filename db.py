from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

engine = create_engine("sqlite:///app.db", future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)
Base = declarative_base()

class Capture(Base):
    __tablename__ = "captures"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=datetime.utcnow)
    raw_path = Column(String)
    det_path = Column(String)
    defect_count = Column(Integer, default=0)
    sample_count = Column(Integer, default=0)
    max_conf = Column(Float, default=0.0)
    note = Column(String, default="")

Base.metadata.create_all(engine)
