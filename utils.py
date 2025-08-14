from db import SessionLocal, Capture

def save_capture_to_db(raw_path, det_path, counts, res):
    try:
        max_conf = float(res.boxes.conf.max().item()) if (res and res.boxes is not None and res.boxes.conf is not None) else 0.0
    except Exception:
        max_conf = 0.0

    with SessionLocal() as s:
        row = Capture(raw_path=raw_path,
                      det_path=det_path,
                      defect_count=int(counts.get("defect", 0)),
                      sample_count=int(counts.get("sample", 0)),
                      max_conf=max_conf)
        s.add(row)
        s.commit()
        return row.id, max_conf
