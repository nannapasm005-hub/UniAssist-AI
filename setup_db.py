"""
สร้าง uniassist.db จาก CSV รายชื่อนักศึกษา
รันครั้งเดียวก่อนเริ่มใช้งาน
"""
import sqlite3
import pandas as pd
import glob

CSV_FILES  = glob.glob("data2/students*.csv")
DB_PATH    = "uniassist.db"

def setup():
    conn = sqlite3.connect(DB_PATH)

    # โหลด CSV ทุกไฟล์รวมกัน
    all_dfs = []
    for f in sorted(CSV_FILES):
        df = pd.read_csv(f, encoding="utf-8-sig")
        all_dfs.append(df)
        print(f"  loaded {f}: {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)

    # สร้างตาราง STUDENTS
    conn.execute("DROP TABLE IF EXISTS STUDENTS")
    conn.execute("""
        CREATE TABLE STUDENTS (
            student_id  TEXT PRIMARY KEY,
            t_prename   TEXT,
            t_name      TEXT,
            t_surname   TEXT,
            e_prename   TEXT,
            e_name      TEXT,
            e_surname   TEXT,
            curriculum  TEXT,
            status      TEXT,
            password    TEXT
        )
    """)

    combined.to_sql("STUDENTS", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"\n✅ uniassist.db: {len(combined)} students")

if __name__ == "__main__":
    if not CSV_FILES:
        print("⚠️ ไม่พบไฟล์ students*.csv")
    else:
        setup()
