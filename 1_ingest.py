"""
STEP 1 - INGEST DATA (v2 — ปรับปรุงจาก v1)
============================================
การเปลี่ยนแปลงหลักจาก v1:
  1. สร้าง document text แบบ "focused" แทนการยุบทุก field เป็น string เดียว
     → แต่ละ record จะถูก embed จาก condition + consequence เป็นหลัก
       ทำให้ vector ที่ได้ "ตรงประเด็น" มากขึ้น ไม่ถูก dilute จาก metadata

  2. แยก prerequisite ออกไปเก็บใน SQLite เป็น structured graph
     → ลักษณะของ prerequisite คือ "A ต้องก่อน B" ซึ่งเป็น relational data
       ไม่เหมาะกับ semantic search เลย ควร lookup ตรง ๆ แทน

  3. เพิ่ม metadata field ที่จำเป็น (year_level, credit_threshold ฯลฯ)
     → ช่วยให้ filter ก่อน retrieve ได้ ลด noise ในผลลัพธ์

  4. เก็บ source_text แยกจาก embedding text
     → source_text เอาไว้ให้ LLM อ้างอิง ไม่ใช่เอาไว้ embed
"""

import json
import glob
import sqlite3
import re
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ===== CONFIG =====
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

JSON_FILES      = glob.glob("data2/*_filtered_labeled.json")
CSV_FILES       = glob.glob("data2/teach_table*.csv")

CHROMA_PATH     = "./chroma_db"
SQLITE_PATH     = "./teach_table.db"
COLLECTION_NAME = "curriculum_rules"
# ==================


# ──────────────────────────────────────────────────────────────
# HELPER: สร้าง document text แบบ "focused"
# ──────────────────────────────────────────────────────────────
def build_focused_text(record: dict) -> str:
    """
    v1 รวมทุก field ด้วย ' | ' ทำให้ embedding vector เป็น "ค่าเฉลี่ย" ของทุกอย่าง
    v2 เน้นที่ condition + consequence ซึ่งคือ "เนื้อหาจริง" ของกฎแต่ละข้อ
    ส่วน branch และ rule_type ใส่ไว้เป็น prefix สั้น ๆ เพื่อช่วย disambiguation
    source_text ถูกตัดออกจาก embed text → ย้ายไปเก็บใน metadata แทน
    """
    rule_type      = record.get("rule_type", "")
    branch_name    = record.get("branch_name_th", "")
    description    = record.get("description", "").strip()
    condition      = (record.get("condition") or "").strip()
    consequence    = (record.get("consequence") or "").strip()

    # สร้าง text ที่กระชับและตรงประเด็น
    parts = [f"[{branch_name} / {rule_type}]"]

    if description:
        parts.append(description)
    if condition:
        parts.append(f"เงื่อนไข: {condition}")
    if consequence:
        parts.append(f"ผลลัพธ์: {consequence}")

    return " ".join(parts)


# ──────────────────────────────────────────────────────────────
# HELPER: แยก prerequisite records ออกจาก batch หลัก
# ──────────────────────────────────────────────────────────────
def extract_prerequisite_rows(record: dict) -> list[dict]:
    """
    แก้ไขจาก ingest_v2:
      - v2 เดิม: ดึงจาก 'prerequisite_codes' (ไม่มีใน JSON จริง) แล้ว fallback ไป 'condition'
      - v3 นี้:  ดึงจาก 'prerequisite_of' ซึ่งเป็น field จริงใน JSON

    กรณีพิเศษที่ต้องจัดการ:
      1. prerequisite_of = "null" (string)  → ไม่มี prerequisite จริง ข้ามไป
      2. prerequisite_of = "A,B"           → prerequisite แบบ OR เช่น BIT
                                             แตกเป็น 2 edges: (subject, A) และ (subject, B)
    """
    rows = []
    subject_code   = str(record.get("subject_code") or "").strip()
    branch         = record.get("branch", "")
    prerequisite_of = record.get("prerequisite_of") or ""

    # กรณีที่ไม่มี prerequisite จริง
    if not prerequisite_of or str(prerequisite_of).strip().lower() == "null":
        return []

    # แตก prerequisite ที่อาจมีหลายตัวคั่นด้วย comma เช่น "06036119,06036122"
    prereq_codes = [
        code.strip()
        for code in str(prerequisite_of).split(",")
        if code.strip() and code.strip().lower() != "null"
    ]

    for prereq_code in prereq_codes:
        if subject_code and prereq_code:
            rows.append({
                "branch":        branch,
                "subject_code":  subject_code,
                "requires_code": prereq_code,
            })

    return rows


# ──────────────────────────────────────────────────────────────
# 1. JSON → ChromaDB  (กฎที่ไม่ใช่ prerequisite)
#    + JSON → SQLite  (prerequisite graph)
# ──────────────────────────────────────────────────────────────
def ingest_json():
    print("📚 Ingesting JSON → ChromaDB + prerequisite graph → SQLite ...")

    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client       = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    docs, metadatas, ids = [], [], []
    prereq_rows = []  # สำหรับ SQLite prerequisite_graph table

    for filepath in JSON_FILES:
        with open(filepath, encoding="utf-8") as f:
            records = json.load(f)

        for record in records:
            rule_type = record.get("rule_type", "")

            # ── Prerequisite → SQLite graph (ไม่ embed) ──────────────
            if rule_type == "prerequisite":
                rows = extract_prerequisite_rows(record)
                prereq_rows.extend(rows)
                # ยังคง embed ไว้ด้วยเพื่อให้ chatbot อธิบายได้
                # แต่ LLM ที่ต้องการ "ลำดับ" จะ lookup จาก SQLite แทน

            # ── Document text: focused, ไม่มี source_text ────────────
            doc_text = build_focused_text(record)
            if not doc_text.strip():
                continue  # ข้ามถ้าไม่มีเนื้อหาจริง

            # ── Metadata: เพิ่ม field ที่ขาดไปใน v1 ──────────────────
            # year_level และ credit_threshold ช่วยให้ filter ก่อน retrieve ได้
            # source_text ย้ายมาอยู่ที่นี่ — ให้ LLM อ้างอิงแต่ไม่ embed
            metadata = {
                "rule_id":          record.get("rule_id", ""),
                "rule_type":        rule_type,
                "branch":           record.get("branch", ""),
                "branch_name_th":   record.get("branch_name_th", ""),
                "subject_code":     str(record.get("subject_code") or ""),
                "gt_confidence":    str(record.get("gt_confidence", "")),
                "source_file":      record.get("_source_file", ""),
                "source_page":      str(record.get("_source_page", "")),
                # ── field ใหม่ที่เพิ่มใน v2 ──
                # year_level: ชั้นปีที่กฎนี้เกี่ยวข้อง (1-4, หรือ 0 = ทุกปี)
                "year_level":       str(record.get("year_level") or "0"),
                # credit_threshold: หน่วยกิตขั้นต่ำที่เกี่ยวข้องกับกฎ
                "credit_threshold": str(record.get("credit_threshold") or "0"),
                # semester_applicable: เทอมที่กฎนี้ใช้ได้ (1, 2, หรือ 0 = ทุกเทอม)
                "semester_applicable": str(record.get("semester_applicable") or "0"),
                # source_text เก็บไว้ให้ LLM อ้างอิง ไม่ใช่สำหรับ embed
                "source_text":      str(record.get("source_text", ""))[:500],
            }

            docs.append(doc_text)
            metadatas.append(metadata)
            ids.append(record.get("_gt_id",
                       f"{record.get('branch')}_{record.get('rule_id')}"))

    # Upsert เข้า ChromaDB เป็น batch
    BATCH = 500
    for i in range(0, len(docs), BATCH):
        collection.upsert(
            documents=docs[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
            ids=ids[i:i+BATCH],
        )
        print(f"  [ChromaDB] upserted {min(i+BATCH, len(docs))}/{len(docs)}")

    print(f"✅ ChromaDB: {collection.count()} records in '{COLLECTION_NAME}'")

    # บันทึก prerequisite graph ลง SQLite
    _save_prereq_graph(prereq_rows)


def _save_prereq_graph(rows: list[dict]):
    """
    บันทึก prerequisite เป็นตาราง relational ใน SQLite
    ตัวอย่าง query ที่ทำได้:
      SELECT requires_code FROM prerequisite_graph
      WHERE subject_code = '06016205' AND branch = 'IT'
    ซึ่งตรงและรวดเร็วกว่าการ semantic search มาก
    """
    if not rows:
        print("  ℹ️  ไม่พบข้อมูล prerequisite ที่ parse ได้")
        return

    conn = sqlite3.connect(SQLITE_PATH)
    df   = pd.DataFrame(rows).drop_duplicates()
    df.to_sql("prerequisite_graph", conn, if_exists="replace", index=False)

    # สร้าง index เพื่อให้ lookup เร็ว
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prereq_subject "
                 "ON prerequisite_graph(subject_code, branch)")
    conn.commit()
    conn.close()
    print(f"✅ SQLite prerequisite_graph: {len(df)} edges")


# ──────────────────────────────────────────────────────────────
# 2. CSV → SQLite  (ตารางเรียน — ไม่เปลี่ยนจาก v1 มากนัก)
# ──────────────────────────────────────────────────────────────
def ingest_csv():
    print("📊 Ingesting CSV → SQLite ...")

    conn    = sqlite3.connect(SQLITE_PATH)
    all_dfs = []

    for filepath in CSV_FILES:
        df = pd.read_csv(filepath, encoding="utf-8-sig")

        sem_match = re.search(r"s?(\d+)\.csv$", filepath)
        semester  = int(sem_match.group(1)) if sem_match else 0
        df["semester"] = semester

        all_dfs.append(df)
        print(f"  loaded {filepath}: {len(df)} rows (semester={semester})")

    combined = pd.concat(all_dfs, ignore_index=True)

    def strip_html(val):
        if pd.isna(val):
            return ""
        return re.sub(r"<[^>]+>", " ", str(val)).strip()

    for col in ["teacher_list_th", "teacher_list_en", "rules_th", "rules_en"]:
        if col in combined.columns:
            combined[col] = combined[col].apply(strip_html)

    combined.to_sql("teach_table", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ SQLite teach_table: {len(combined)} rows\n")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not JSON_FILES:
        print("⚠️  ไม่พบไฟล์ *_filtered_labeled.json")
    else:
        ingest_json()

    if not CSV_FILES:
        print("⚠️  ไม่พบไฟล์ teach_table*.csv")
    else:
        ingest_csv()

    print("\n🎉 Ingest v2 เสร็จสิ้น!")
    print("   สิ่งที่เปลี่ยนไปจาก v1:")
    print("   - ChromaDB embed จาก focused text (condition+consequence)")
    print("   - prerequisite_graph table ใน SQLite สำหรับ structured lookup")
    print("   - metadata มี year_level, credit_threshold, semester_applicable")
    print("   - source_text อยู่ใน metadata ไม่ถูก embed แล้ว")