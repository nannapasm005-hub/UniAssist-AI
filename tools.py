"""
STEP 2 - TOOLS (v2)
====================
การเปลี่ยนแปลงจาก v1:

  1. search_curriculum_rules  → เพิ่ม rule_type filter + แสดง source_text ใน output
  2. query_teach_table        → ไม่เปลี่ยน (ทำงานได้ดีอยู่แล้ว)
  3. simulate_gpa_scenario    → ไม่เปลี่ยน
  4. query_prerequisite_graph  → lookup ตาราง SQLite ที่ ingest_v2 สร้างไว้
  5. recommend_courses         → รวม prerequisite check + ตารางเรียน เข้าด้วยกัน
"""

import sqlite3
import chromadb
from langchain.tools import tool
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ===== CONFIG =====
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_PATH     = "./chroma_db"
SQLITE_PATH      = "./teach_table.db"       # ตารางเรียน + prerequisite
PLAN_SQLITE_PATH = "./plan_teach_table.db"  # แผนการเรียน (study_plan)
COLLECTION_NAME = "curriculum_rules"
# ==================


def _get_collection():
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME, embedding_function=embedding_fn)


# ──────────────────────────────────────────────────────────────
# TOOL 1: search_curriculum_rules (v2 — เพิ่ม rule_type filter)
# ──────────────────────────────────────────────────────────────
@tool
def search_curriculum_rules(query: str, branch: str = "", rule_type: str = "") -> str:
    """
    ค้นหากฎระเบียบและข้อมูลหลักสูตรจาก Vector Database
    ใช้เมื่อผู้ใช้ถามเกี่ยวกับ:
    - กฎการลงทะเบียน / prerequisite
    - เงื่อนไขการพ้นสภาพ / วิทยาทัณฑ์
    - จำนวนหน่วยกิต / โครงสร้างหลักสูตร
    - ระเบียบการศึกษาทั่วไป
    - เกียรตินิยม (honors) / เงื่อนไขสำเร็จการศึกษา (graduation)

    Args:
        query:     คำถามหรือ keyword ที่ต้องการค้นหา
        branch:    ระบุสาขาถ้าทราบ เช่น 'AIT', 'IT', 'BIT', 'DSBA'
                   (ถ้าไม่ระบุจะค้นทุกสาขา)
        rule_type: กรองเฉพาะประเภทกฎ — ควรระบุเสมอถ้าทราบ เพื่อลด noise
                   ค่าที่ใช้ได้: 'registration', 'prerequisite', 'dismissal',
                                 'gpa', 'graduation', 'honors', 'assessment'
    """
    try:
        collection = _get_collection()

        # สร้าง filter จาก branch และ rule_type ที่ระบุมา
        # v1 filter แค่ branch → ดึง rule type อื่นปนมาด้วย
        # v2 filter ทั้งคู่ถ้ามี → ลด noise ลงได้มาก
        conditions = {}
        if branch:
            conditions["branch"] = branch
        if rule_type:
            conditions["rule_type"] = rule_type

        where = {"$and": [
            {k: v} for k, v in conditions.items()
        ]} if len(conditions) > 1 else (conditions if conditions else None)

        results = collection.query(
            query_texts=[query],
            n_results=5,
            where=where,
        )

        docs   = results["documents"][0]
        metas  = results["metadatas"][0]

        if not docs:
            # ถ้า filter เข้มงวดเกินไปจนไม่เจอ ให้ลอง fallback โดยไม่ filter rule_type
            if rule_type:
                fallback_where = {"branch": branch} if branch else None
                results = collection.query(
                    query_texts=[query],
                    n_results=5,
                    where=fallback_where,
                )
                docs  = results["documents"][0]
                metas = results["metadatas"][0]

            if not docs:
                return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลหลักสูตร"

        output = []
        for i, (doc, meta) in enumerate(zip(docs, metas), 1):
            # v2: เพิ่ม source_text ใน output เพื่อให้ LLM อ้างอิงข้อความต้นฉบับได้
            source = meta.get("source_text", "")
            source_line = f"ต้นฉบับ: {source}" if source else ""

            output.append(
                f"[{i}] สาขา: {meta.get('branch_name_th')} | "
                f"ประเภท: {meta.get('rule_type')} | "
                f"rule_id: {meta.get('rule_id')}\n"
                f"{doc}"
                + (f"\n{source_line}" if source_line else "")
            )

        return "\n\n".join(output)

    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหา: {str(e)}"


# ──────────────────────────────────────────────────────────────
# TOOL 2: query_teach_table (ไม่เปลี่ยนจาก v1)
# ──────────────────────────────────────────────────────────────
@tool
def query_teach_table(sql: str) -> str:
    """
    ค้นหาข้อมูลตารางเรียนจากฐานข้อมูล SQLite ด้วย SQL query
    ใช้เมื่อผู้ใช้ถามเกี่ยวกับ:
    - รายวิชาที่เปิดสอน / เวลาเรียน / ห้องเรียน
    - อาจารย์ผู้สอน
    - section ที่ยังมีที่ว่าง
    - วันสอบ midterm / final
    - เงื่อนไขการลงทะเบียน (rules_th)

    Schema ของตาราง 'teach_table':
      - curriculum_name_th       : ชื่อหลักสูตร มีค่าดังนี้เท่านั้น:
                                   'เทคโนโลยีสารสนเทศ'
                                   'เทคโนโลยีปัญญาประดิษฐ์'
                                   'วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ'
                                   'เทคโนโลยีสารสนเทศทางธุรกิจ (นานาชาติ)'
      - class_year               : ชั้นปี (1-4)
      - subject_id               : รหัสวิชา
      - subject_name_th          : ชื่อวิชาภาษาไทย
      - subject_name_en          : ชื่อวิชาภาษาอังกฤษ
      - credit                   : จำนวนหน่วยกิต
      - section                  : หมู่เรียน
      - teach_day                : วันที่สอน (1=จันทร์ ... 7=อาทิตย์)
      - teach_time               : เวลาเริ่ม (HH:MM:SS)
      - teach_time2              : เวลาสิ้นสุด (HH:MM:SS)
      - classroom                : ห้องเรียน
      - teacher_list_th          : รายชื่ออาจารย์ (ภาษาไทย)
      - midterm_start_date_time  : วันสอบกลางภาค
      - final_start_date_time    : วันสอบปลายภาค
      - limit                    : จำนวนที่นั่งสูงสุด
      - count                    : จำนวนที่ลงทะเบียนแล้ว
      - rules_th                 : เงื่อนไขการลงทะเบียน
      - semester                 : เทอมที่เปิดสอน (1=เทอม 1, 2=เทอม 2)

    Args:
        sql: SQL query (SELECT เท่านั้น ห้าม INSERT/UPDATE/DELETE)
    """
    try:
        forbidden = ["insert", "update", "delete", "drop", "alter", "create"]
        if any(word in sql.lower() for word in forbidden):
            return "⛔ ไม่อนุญาตให้แก้ไขข้อมูล อนุญาตเฉพาะ SELECT เท่านั้น"

        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows   = cursor.fetchall()
        conn.close()

        if not rows:
            return "ไม่พบข้อมูลที่ตรงกับเงื่อนไข"

        cols = rows[0].keys()
        result_lines = [" | ".join(cols)]
        result_lines.append("-" * 80)
        for row in rows[:20]:
            result_lines.append(" | ".join(str(row[c]) for c in cols))

        if len(rows) > 20:
            result_lines.append(f"... และอีก {len(rows) - 20} รายการ")

        return "\n".join(result_lines)

    except Exception as e:
        return f"เกิดข้อผิดพลาดใน SQL query: {str(e)}\nSQL ที่ใช้: {sql}"


# ──────────────────────────────────────────────────────────────
# TOOL 3: simulate_gpa_scenario (ไม่เปลี่ยนจาก v1)
# ──────────────────────────────────────────────────────────────
@tool
def simulate_gpa_scenario(
    current_gpax: float,
    credits_earned: int,
    remaining_sems: int,
    credits_per_sem: int,
    sem_gpas: str,
    credits_remaining: int = None,
) -> str:
    """
    จำลอง scenario GPA ว่าถ้าได้ GPA ตามที่สมมติแต่ละเทอม GPAX จะเป็นเท่าไหร่
    และต้องทำอีกเท่าไหร่เพื่อบรรลุเป้าหมาย (Honors, พ้น Probation)
    ใช้เมื่อผู้ใช้ถามว่า "ถ้าได้ X เทอมนี้ จะได้ GPAX เท่าไหร่" หรือ "เหลือกี่เทอมถึงจะได้เกียรตินิยม"

    Args:
        current_gpax:      GPAX ปัจจุบัน เช่น 3.67
        credits_earned:    หน่วยกิตสะสม เช่น 89
        remaining_sems:    จำนวนเทอมที่เหลือ เช่น 2
        credits_per_sem:   หน่วยกิตที่จะลงต่อเทอม เช่น 18
        sem_gpas:          GPA ที่สมมติแต่ละเทอม คั่นด้วยจุลภาค เช่น "4.0,3.5"
        credits_remaining: หน่วยกิตที่เหลือจริงทั้งหมด เช่น 31
    """
    try:
        from grade_module import simulate_scenarios, format_scenario_text
        gpas   = [float(x.strip()) for x in sem_gpas.split(",") if x.strip()]
        result = simulate_scenarios(
            current_gpax=current_gpax,
            credits_earned=credits_earned,
            remaining_sems=remaining_sems,
            credits_per_sem=credits_per_sem,
            sem_gpas=gpas,
            total_credits_remaining=credits_remaining,
        )
        return format_scenario_text(result)
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


# ──────────────────────────────────────────────────────────────
# TOOL 4: [ใหม่] query_prerequisite_graph
# ──────────────────────────────────────────────────────────────
@tool
def query_prerequisite_graph(subject_code: str, branch: str = "") -> str:
    """
    ตรวจสอบ prerequisite (วิชาบังคับก่อน) ของวิชาที่ระบุ
    โดย lookup ตรงจากตาราง prerequisite_graph ใน SQLite

    ใช้แทน search_curriculum_rules สำหรับคำถาม prerequisite โดยเฉพาะ
    เพราะ prerequisite เป็นข้อมูลเชิง relational (A ต้องก่อน B)
    การ lookup ตรงจาก SQLite จึงถูกต้อง 100% และเร็วกว่า semantic search

    ใช้เมื่อผู้ใช้ถามว่า:
    - "วิชา X ต้องเรียนวิชาอะไรก่อน?"
    - "ฉันเรียน A และ B แล้ว ลงวิชา C ได้เลยไหม?"
    - "prerequisite ของ [รหัสวิชา] คืออะไร?"

    Args:
        subject_code: รหัสวิชาที่ต้องการตรวจสอบ เช่น '06016205'
        branch:       สาขา เช่น 'AIT', 'IT' (optional)
    """
    try:
        conn = sqlite3.connect(SQLITE_PATH)

        # ตรวจสอบก่อนว่าตารางมีอยู่จริง (ingest_v2 เป็นคนสร้าง)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t[0] for t in tables]

        if "prerequisite_graph" not in table_names:
            conn.close()
            return (
                "ไม่พบตาราง prerequisite_graph ในฐานข้อมูล "
                "กรุณารัน ingest_v2.py ก่อนเพื่อสร้างตารางนี้"
            )

        # Query หา prerequisite ของวิชาที่ถาม
        if branch:
            rows = conn.execute(
                "SELECT subject_code, requires_code, branch "
                "FROM prerequisite_graph "
                "WHERE subject_code = ? AND branch = ?",
                (subject_code, branch)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT subject_code, requires_code, branch "
                "FROM prerequisite_graph "
                "WHERE subject_code = ?",
                (subject_code,)
            ).fetchall()

        # Query กลับด้าน: วิชาไหนบ้างที่ต้องใช้วิชานี้เป็น prerequisite
        dependents = conn.execute(
            "SELECT subject_code FROM prerequisite_graph WHERE requires_code = ?",
            (subject_code,)
        ).fetchall()

        conn.close()

        if not rows:
            return (
                f"วิชา {subject_code} ไม่มี prerequisite "
                f"(สามารถลงทะเบียนได้โดยไม่ต้องเรียนวิชาใดก่อน)"
            )

        prereq_codes = [r[1] for r in rows]
        branch_info  = rows[0][2] if rows else ""

        output = [
            f"วิชา {subject_code} (สาขา {branch_info}) "
            f"มี prerequisite ดังนี้:"
        ]
        for code in prereq_codes:
            output.append(f"  - ต้องผ่านวิชา {code} ก่อน")

        if dependents:
            dep_codes = [d[0] for d in dependents]
            output.append(
                f"\nวิชาที่ใช้ {subject_code} เป็น prerequisite: "
                + ", ".join(dep_codes)
            )

        return "\n".join(output)

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


# ──────────────────────────────────────────────────────────────
# TOOL 5: [ใหม่] recommend_courses
# ──────────────────────────────────────────────────────────────
@tool
def recommend_courses(
    branch: str,
    year_level: int,
    semester: int,
    passed_subject_codes: str = "",
    max_credits: int = 22,
) -> str:
    """
    แนะนำรายวิชาที่นักศึกษาสามารถลงทะเบียนได้ในเทอมนี้
    โดยตรวจสอบ prerequisite, ตารางเรียน, และภาระหน่วยกิต อัตโนมัติ

    ใช้เมื่อผู้ใช้ถามว่า:
    - "เทอมนี้ควรลงวิชาอะไรบ้าง?"
    - "ฉันเรียนมาแล้ว X, Y ลงวิชาอะไรได้อีก?"
    - "ปี 2 เทอม 1 สาขา AIT ควรเรียนวิชาอะไร?"

    Args:
        branch:               สาขา เช่น 'AIT', 'IT', 'BIT', 'DSBA'
        year_level:           ชั้นปีของนักศึกษา (1-4)
        semester:             เทอมที่ต้องการแนะนำ (1 หรือ 2)
        passed_subject_codes: รหัสวิชาที่ผ่านมาแล้ว คั่นด้วย comma
                              เช่น '06016101,06016102,06016201'
        max_credits:          หน่วยกิตสูงสุดที่ต้องการลง (default 22)
    """
    try:
        conn = sqlite3.connect(SQLITE_PATH)

        # แปลง branch → ชื่อหลักสูตรในตาราง
        branch_map = {
            "AIT":  "เทคโนโลยีปัญญาประดิษฐ์",
            "IT":   "เทคโนโลยีสารสนเทศ",
            "DSBA": "วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ",
            "BIT":  "เทคโนโลยีสารสนเทศทางธุรกิจ (นานาชาติ)",
        }
        curriculum_name = branch_map.get(branch.upper(), branch)

        # รายวิชาที่ผ่านมาแล้ว
        passed_codes = set()
        if passed_subject_codes.strip():
            passed_codes = {
                c.strip() for c in passed_subject_codes.split(",") if c.strip()
            }

        # ดึงวิชาทั้งหมดที่เปิดสอนในเทอมและชั้นปีที่ต้องการ
        rows = conn.execute(
            """
            SELECT DISTINCT subject_id, subject_name_th, credit,
                   teach_day, teach_time, teach_time2, classroom,
                   section, rules_th
            FROM teach_table
            WHERE curriculum_name_th = ?
              AND class_year = ?
              AND semester = ?
            ORDER BY subject_id
            """,
            (curriculum_name, year_level, semester)
        ).fetchall()

        if not rows:
            conn.close()
            return (
                f"ไม่พบรายวิชาสำหรับ {branch} ปี {year_level} เทอม {semester} "
                f"ในฐานข้อมูล กรุณาตรวจสอบข้อมูลตารางเรียน"
            )

        # ตรวจสอบ prerequisite จาก prerequisite_graph (ถ้ามี)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        has_prereq_table = "prerequisite_graph" in [t[0] for t in tables]

        eligible   = []  # ลงได้เลย
        blocked    = []  # prerequisite ยังไม่ผ่าน
        total_credits = 0

        for row in rows:
            subject_id   = str(row[0])
            subject_name = row[1]
            credit       = row[2] or 0

            # ข้ามวิชาที่ผ่านมาแล้ว
            if subject_id in passed_codes:
                continue

            # ตรวจ prerequisite
            missing_prereqs = []
            if has_prereq_table:
                prereq_rows = conn.execute(
                    "SELECT requires_code FROM prerequisite_graph "
                    "WHERE subject_code = ? AND branch = ?",
                    (subject_id, branch.upper())
                ).fetchall()
                missing_prereqs = [
                    r[0] for r in prereq_rows if r[0] not in passed_codes
                ]

            day_map  = {1:"จ", 2:"อ", 3:"พ", 4:"พฤ", 5:"ศ", 6:"ส", 7:"อา"}
            day_str  = day_map.get(row[3], str(row[3]))
            time_str = f"{row[4] or '?'[:5]}–{row[5] or '?'[:5]}"

            info = (
                f"  {subject_id} {subject_name} "
                f"({credit} หน่วยกิต) | {day_str} {time_str} | {row[6] or '?'}"
            )

            if missing_prereqs:
                blocked.append(
                    info + f"\n    ⛔ ยังขาด prerequisite: {', '.join(missing_prereqs)}"
                )
            else:
                if total_credits + credit <= max_credits:
                    total_credits += credit
                    eligible.append(info)
                else:
                    eligible.append(info + "  [เกิน max_credits ถ้าลงครบ]")

        conn.close()

        output = [
            f"📚 รายวิชาแนะนำ: {branch} ปี {year_level} เทอม {semester}",
            f"   (วิชาที่ผ่านมาแล้ว {len(passed_codes)} วิชา | "
            f"หน่วยกิตสูงสุด {max_credits})",
            "",
        ]

        if eligible:
            output.append(f"✅ ลงได้เลย ({len(eligible)} วิชา ~{total_credits} หน่วยกิต):")
            output.extend(eligible)
        else:
            output.append("✅ ไม่พบวิชาที่ลงได้ในเทอมนี้")

        if blocked:
            output.append(f"\n⛔ ยังลงไม่ได้ — prerequisite ไม่ครบ ({len(blocked)} วิชา):")
            output.extend(blocked)

        return "\n".join(output)

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"


# ──────────────────────────────────────────────────────────────
# HELPER: ดึง passed/in-progress codes จาก transcript text
# ──────────────────────────────────────────────────────────────
def _extract_codes_from_transcript(transcript_text: str) -> tuple[set, set]:
    """
    ใช้ grade_module.parse_transcript ดึงรหัสวิชาที่ผ่านและกำลังเรียน
    - passed: grade ที่นับว่าผ่าน = A,B+,B,C+,C,D+,D,S,P (รวม Transfer)
    - in_progress: วิชาที่ GPS : - (กำลังเรียนอยู่เทอมนี้)
    - ไม่นับ F, W, U
    """
    from grade_module import parse_transcript
    parsed = parse_transcript(transcript_text)

    PASS_GRADES = {"A", "B+", "B", "C+", "C", "D+", "D", "S", "P"}

    passed = set()
    for cid, records in parsed["courses_raw"].items():
        # ใช้ record ล่าสุดที่ไม่ใช่ W
        valid = [r for r in records if r.grade != "W"]
        if valid:
            latest = valid[-1]
            if latest.grade in PASS_GRADES:
                passed.add(cid)

    in_progress = {c["course_id"] for c in parsed.get("in_progress", [])}

    return passed, in_progress


# ──────────────────────────────────────────────────────────────
# TOOL 6: check_graduation_eligibility
# ──────────────────────────────────────────────────────────────
@tool
def check_graduation_eligibility(
    branch: str,
    passed_subject_codes: str,
    plan_type: str = "coop",
    in_progress_codes: str = "",
) -> str:
    """
    ตรวจสอบว่านักศึกษาสำเร็จการศึกษาได้หรือยัง

    หลักการ:
    - วิชาบังคับทุกวิชาในแผนต้องผ่านครบ ห้ามขาดแม้แต่วิชาเดียว
    - วิชาเลือก (subject_code มี x หรือ = ELECTIVE) เช็คแค่หน่วยกิตรวม
    - สหกิจ: วิชาในประเทศ/ต่างประเทศ เลือกผ่านอันใดอันหนึ่งก็ได้
    - วิชากำลังเรียน (in_progress_codes) แสดงแยก ไม่นับว่าขาด

    Args:
        branch:               สาขา เช่น 'AIT', 'IT', 'BIT', 'DSBA'
        passed_subject_codes: รหัสวิชาที่ผ่านแล้วทั้งหมด คั่นด้วย comma
                              ดึงจาก [passed_codes สำหรับ tool] ใน message
        plan_type:            'coop' (default) หรือ 'normal'
        in_progress_codes:    รหัสวิชากำลังเรียนอยู่ คั่นด้วย comma
                              ดึงจาก [in_progress_codes สำหรับ tool] ใน message
    """
    try:
        passed      = {c.strip() for c in passed_subject_codes.split(",") if c.strip()} if passed_subject_codes.strip() else set()
        in_progress = {c.strip() for c in in_progress_codes.split(",") if c.strip()} if in_progress_codes.strip() else set()

        conn = sqlite3.connect(PLAN_SQLITE_PATH)

        tables = [t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        if "study_plan" not in tables:
            conn.close()
            return "ไม่พบตาราง study_plan ในฐานข้อมูล กรุณารัน import_study_plan.py ก่อน"

        plan_type_filter = plan_type if plan_type in ("normal", "coop") else "coop"
        all_plan_rows = conn.execute(
            """
            SELECT subject_code, subject_name_th, year, semester, credits
            FROM study_plan
            WHERE branch = ? AND plan_type = ?
            ORDER BY year, semester, subject_code
            """,
            (branch.upper(), plan_type_filter)
        ).fetchall()

        if not all_plan_rows:
            all_plan_rows = conn.execute(
                """
                SELECT subject_code, subject_name_th, year, semester, credits
                FROM study_plan WHERE branch = ?
                ORDER BY year, semester, subject_code
                """,
                (branch.upper(),)
            ).fetchall()

        conn.close()

        if not all_plan_rows:
            return f"ไม่พบแผนการเรียนของสาขา {branch}"

        # ── OR groups (สหกิจในประเทศ vs ต่างประเทศ) ─────────────────────────
        OR_GROUPS = [
            frozenset({"06046443", "06046444"}),
            frozenset({"06026259", "06026260"}),
            frozenset({"06016481", "06016482"}),
            frozenset({"06036147", "06036148"}),
        ]

        def is_elective(code: str) -> bool:
            """free elective = ELECTIVE หรือ x ทั้งหมด (ไม่มีตัวเลขนำหน้า 4 ตัวขึ้นไป)"""
            c = code.upper()
            if c == "ELECTIVE":
                return True
            digits = sum(1 for ch in c if ch.isdigit())
            return digits < 4 and 'X' in c

        def has_x(code: str) -> bool:
            return 'x' in code or 'X' in code

        def find_or_group(code: str):
            for grp in OR_GROUPS:
                if code in grp:
                    return grp
            return None

        # แยก 3 กลุ่ม
        free_elective_slots      = [(c,n,y,s,cr) for c,n,y,s,cr in all_plan_rows if is_elective(c)]
        mandatory_elective_rows  = [(c,n,y,s,cr) for c,n,y,s,cr in all_plan_rows if has_x(c) and not is_elective(c)]
        pure_required_rows       = [(c,n,y,s,cr) for c,n,y,s,cr in all_plan_rows if not has_x(c)]
        pure_required_codes      = {c for c,*_ in pure_required_rows}

        # ── เทียบวิชาบังคับ (pure required เท่านั้น) ─────────────────────────
        required_rows = pure_required_rows

        # ── เทียบวิชาบังคับ ──────────────────────────────────────────────────
        completed    = []
        in_prog_list = []
        missing      = []
        skipped_or   = set()

        for code, name_th, year, semester, credits in required_rows:
            or_grp = find_or_group(code)

            if or_grp and or_grp in skipped_or:
                continue

            if code in passed:
                completed.append((code, name_th, year, semester, credits))
                if or_grp:
                    skipped_or.add(or_grp)
            elif code in in_progress:
                in_prog_list.append((code, name_th, year, semester, credits))
                if or_grp:
                    skipped_or.add(or_grp)
            elif or_grp:
                if or_grp & (passed | in_progress):
                    skipped_or.add(or_grp)
                    continue
                else:
                    if or_grp not in skipped_or:
                        or_names = " หรือ ".join(sorted(or_grp))
                        missing.append((or_names, "สหกิจ (เลือกอันใดอันหนึ่ง)", year, semester, credits))
                        skipped_or.add(or_grp)
            else:
                missing.append((code, name_th, year, semester, credits))

        # ── เช็ควิชาเลือกบังคับกลุ่ม (060464xx, 060164xx, etc.) ─────────────
        from collections import defaultdict
        me_groups = defaultdict(list)
        for c, n, y, s, cr in mandatory_elective_rows:
            prefix = c.rstrip('xX')
            me_groups[prefix].append((c, n, y, s, cr))

        # รวม passed + in_progress เพื่อคำนวณ elective ที่ลงแล้ว
        passed_and_inprog = passed | in_progress

        mandatory_elective_missing = []
        for prefix, rows_in_group in me_groups.items():
            credits_needed = sum(cr for _, _, _, _, cr in rows_in_group)
            done_in_group = [c for c in passed_and_inprog
                             if c.startswith(prefix) and c not in pure_required_codes]
            passed_in_group = [c for c in passed
                               if c.startswith(prefix) and c not in pure_required_codes]
            inprog_in_group = [c for c in in_progress
                               if c.startswith(prefix) and c not in pure_required_codes]
            credits_done = len(passed_in_group) * 3
            credits_inprog = len(inprog_in_group) * 3
            credits_total = credits_done + credits_inprog
            if credits_total < credits_needed:
                first = rows_in_group[0]
                name_display = first[1].split("(")[0].strip()
                note = f"ผ่านแล้ว {credits_done}" + (f", กำลังเรียน {credits_inprog}" if credits_inprog else "") + f", ต้องครบ {credits_needed} หน่วยกิต"
                mandatory_elective_missing.append(
                    (first[0], f"{name_display} ({note})",
                     first[2], first[3], credits_needed - credits_total)
                )

        missing = missing + mandatory_elective_missing

        # ── เช็ค free elective (ELECTIVE / xxxxxxxx) ────────────────────────
        elective_credits_required = sum(cr for _, _, _, _, cr in free_elective_slots)
        mandatory_elective_prefixes = list(me_groups.keys())
        passed_free_electives = [
            c for c in passed
            if c not in pure_required_codes
            and not any(c.startswith(pfx) for pfx in mandatory_elective_prefixes)
        ]
        elective_credits_done   = len(passed_free_electives) * 3
        elective_ok             = elective_credits_done >= elective_credits_required
        elective_deficit        = max(0, elective_credits_required - elective_credits_done)

        # ── สรุปผล ────────────────────────────────────────────────────────────
        plan_label = "สหกิจศึกษา" if plan_type_filter == "coop" else "ปกติ"
        all_ok = len(missing) == 0 and elective_ok

        output = [
            f"📋 ผลการตรวจสอบเงื่อนไขสำเร็จการศึกษา — สาขา {branch} (แผน{plan_label})",
            f"วิชาบังคับผ่านแล้ว: {len(completed)} วิชา",
            f"วิชาบังคับที่ยังขาด: {len(missing)} รายการ",
            f"วิชาที่กำลังเรียนเทอมนี้: {len(in_prog_list)} วิชา (ยังไม่นับเป็นผ่าน)",
            "",
        ]

        if in_prog_list:
            output.append("📚 วิชาที่ลงทะเบียนไว้เทอมนี้ (รอผลเกรด):")
            for code, name_th, year, semester, credits in in_prog_list:
                output.append(f"  • {code}  {name_th}  ({credits} หน่วยกิต)")
            output.append("")

        if completed:
            output.append(f"✅ วิชาบังคับที่ผ่านแล้ว ({len(completed)} วิชา):")
            current_group = None
            for code, name_th, year, semester, credits in completed:
                group = f"ปีที่ {year} เทอม {semester}"
                if group != current_group:
                    output.append(f"  [{group}]")
                    current_group = group
                output.append(f"    ✓ {code}  {name_th}")
            output.append("")

        if all_ok:
            output.append("🎉 ผ่านครบทุกเงื่อนไขในแผนการเรียนแล้ว!")
            output.append("   (ให้ตรวจสอบเงื่อนไขอื่นๆ เพิ่มเติมกับอาจารย์ที่ปรึกษา)")
        else:
            output.append("❌ ยังสำเร็จการศึกษาไม่ได้ เนื่องจาก:")

            if missing:
                output.append(f"\n  วิชาบังคับที่ยังขาด {len(missing)} รายการ:")
                current_group = None
                for code, name_th, year, semester, credits in missing:
                    group = f"ปีที่ {year} เทอม {semester}"
                    if group != current_group:
                        output.append(f"  [{group}]")
                        current_group = group
                    output.append(f"    ✗ {code}  {name_th}  ({credits} หน่วยกิต)")

            if not elective_ok:
                output.append(
                    f"\n  วิชาเลือกยังขาดอีกประมาณ {elective_deficit} หน่วยกิต "
                    f"(ลงวิชาอะไรก็ได้ให้ครบ {elective_credits_required} หน่วยกิต)"
                )

            # mandatory elective summary
            if mandatory_elective_missing:
                output.append("\n  วิชาเลือกบังคับกลุ่ม (ยังขาด):")
                for code, name_th, year, semester, credits in mandatory_elective_missing:
                    output.append(f"    ✗ {code}  {name_th}  ({credits} หน่วยกิต)")

        return "\n".join(output)

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}"



# ──────────────────────────────────────────────────────────────
# Export
# ──────────────────────────────────────────────────────────────
TOOLS = [
    search_curriculum_rules,
    query_teach_table,
    simulate_gpa_scenario,
    query_prerequisite_graph,
    recommend_courses,
    check_graduation_eligibility,  # ใหม่
]