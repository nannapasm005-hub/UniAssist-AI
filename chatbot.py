"""
chatbot.py — wrapper เชื่อม agent.py กับ main.py
"""
from agent import run_agent


def _parse_transcript_to_codes(transcript_text: str) -> dict:
    """
    Pre-parse transcript → passed_codes, in_progress_codes, gpa, credits
    เพื่อไม่ต้องส่ง full transcript เข้า LLM

    detect in_progress 2 วิธี:
    1. grade_module.parse_transcript (ใช้ GPS : -)
    2. regex scan หารหัส 8 หลักที่ไม่มี grade ตามหลัง (รองรับ GPS : 4 หรือ GPS ว่าง)
    """
    import re
    from grade_module import parse_transcript

    PASS_GRADES = {"A", "B+", "B", "C+", "C", "D+", "D", "S", "P"}
    FAIL_GRADES = {"F", "W", "U"}
    ALL_GRADES  = PASS_GRADES | FAIL_GRADES

    parsed = parse_transcript(transcript_text)

    passed = set()
    for cid, records in parsed["courses_raw"].items():
        valid = [r for r in records if r.grade != "W"]
        if valid and valid[-1].grade in PASS_GRADES:
            passed.add(cid)

    # วิธี 1: จาก grade_module
    in_progress = {c["course_id"] for c in parsed.get("in_progress", [])} - passed

    # วิธี 2: scan หา course ID ที่ไม่มี grade เฉพาะเทอมล่าสุด (GPS ไม่ใช่ - หรือไม่มี GPS)
    # แบ่ง transcript เป็น semester blocks แล้วหา block สุดท้ายที่ยังไม่มี GPS จริงๆ
    sem_header = re.compile(
        r'\d+(?:st|nd|rd|th)\s+Semester[^\n]*\d{4}',
        re.IGNORECASE
    )
    blocks = re.split(sem_header, transcript_text)
    headers = sem_header.findall(transcript_text)

    # หา block สุดท้ายที่มีวิชาไม่มี grade
    has_grade_pattern = re.compile(
        r'(\d{8})\s+[A-Za-z][A-Za-z0-9 ,\-&/()\t]+?\s+\d\s+(A|B\+|B|C\+|C|D\+|D|F|S|U|W|P)'
    )
    no_grade_inline = re.compile(
        r'^(\d{8})\s+[A-Za-z][A-Za-z0-9 ,\-&/()\t]+?\s+(\d)\s*$',
        re.MULTILINE
    )

    graded = {m.group(1) for m in has_grade_pattern.finditer(transcript_text)}

    # ดูเฉพาะ block สุดท้าย
    if len(blocks) > 1:
        last_block = blocks[-1]
        for m in no_grade_inline.finditer(last_block):
            cid = m.group(1)
            if cid not in graded and cid not in passed:
                in_progress.add(cid)

    # ลบออกถ้าอยู่ใน passed แล้ว
    in_progress -= passed

    return {
        "passed_codes":      ",".join(sorted(passed)),
        "in_progress_codes": ",".join(sorted(in_progress)),
        "cumulative_gpa":    parsed.get("cumulative_gpa"),
        "credits_earned":    parsed.get("total_credits"),
    }


# ── /api/chat ──────────────────────────────────────────────────────────────────
def chat(message: str, transcript_text: str = None, chat_history: list = None) -> dict:
    """frontend คาดหวัง: { mode: 'curriculum', answer: '...' }"""
    from langchain_core.messages import HumanMessage, AIMessage

    # จำกัด history แค่ 6 messages ล่าสุด
    history = []
    if chat_history:
        for h in chat_history[-6:]:
            if h.get('role') == 'user':
                history.append(HumanMessage(content=h['content']))
            elif h.get('role') == 'assistant':
                history.append(AIMessage(content=h['content']))

    if transcript_text:
        # parse transcript ก่อน แล้วส่งแค่ summary + codes เข้า LLM
        info = _parse_transcript_to_codes(transcript_text)
        transcript_summary = (
            f"GPAX: {info['cumulative_gpa']} | หน่วยกิตสะสม: {info['credits_earned']}\n"
            f"วิชาที่ผ่านแล้ว ({len(info['passed_codes'].split(','))} วิชา): {info['passed_codes']}\n"
            f"วิชากำลังเรียน: {info['in_progress_codes'] or 'ไม่มี'}"
        )
        full_message = (
            f"{message}\n\n"
            f"[ข้อมูล Transcript]\n{transcript_summary}\n\n"
            f"[passed_codes สำหรับ tool]: {info['passed_codes']}\n"
            f"[in_progress_codes สำหรับ tool]: {info['in_progress_codes']}"
        )
    else:
        full_message = message

    answer = run_agent(full_message, history)
    return {"mode": "curriculum", "answer": answer}


# ── /api/analyze-transcript ────────────────────────────────────────────────────
def analyze_transcript(transcript_text: str, student_id: str = None) -> dict:
    """
    frontend คาดหวัง:
    {
        summary: { gpa, credits_earned, credits_total, credits_remaining, failed_courses_count },
        risk:    { level: 'ปกติ'|'เสี่ยง'|'วิกฤต', detail: [...] },
        ai_summary: '...'
    }
    """
    from grade_module import analyze_transcript_full
    from agent import run_llm_direct
    import sqlite3

    # ดึง curriculum ของนักศึกษาจาก DB
    curriculum = None
    if student_id:
        try:
            conn = sqlite3.connect("uniassist.db")
            row  = conn.execute(
                "SELECT curriculum FROM STUDENTS WHERE student_id = ?", (student_id,)
            ).fetchone()
            conn.close()
            if row:
                curriculum = row[0]
        except:
            pass

    result     = analyze_transcript_full(transcript_text, curriculum=curriculum)
    ai_summary = run_llm_direct(result["llm_prompt"])

    return {
        "summary":    result["summary"],
        "risk":       result["risk"],
        "honors":     result["honors"],   # ← เพิ่มบรรทัดนี้
        "ai_summary": ai_summary,
    }