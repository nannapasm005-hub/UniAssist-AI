"""
grade_module.py — Grade Module สำหรับ UniAssist AI
ครอบคลุม Step 1-6 (Step 7 = test ด้านล่าง)

การใช้งาน:
    from grade_module import analyze_transcript_full
    result = analyze_transcript_full(transcript_text, next_semester_credits=18)
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# STEP 1 — RULE ENGINE
# ═══════════════════════════════════════════════════════════════

GRADE_POINTS = {
    "A": 4.0, "B+": 3.5, "B": 3.0, "C+": 2.5,
    "C": 2.0, "D+": 1.5, "D": 1.0, "F": 0.0,
    "S": None, "U": None, "W": None, "P": None,  # ไม่นำมาคำนวณ GPA
}

RULES = {
    "probation":            lambda gpax: gpax < 2.00,
    "near_probation":       lambda gpax: 2.00 <= gpax < 2.25,
    "second_class_honors":  lambda gpax: gpax >= 3.25,
    "first_class_honors":   lambda gpax: gpax >= 3.50,
    "gold_medal_track":     lambda gpax: gpax >= 3.75,
}

# เงื่อนไข Honors เพิ่มเติม (นอกจาก GPAX)
HONORS_MAX_TRANSFER_RATIO = 0.30   # transfer ไม่เกิน 30% ของ total credits
HONORS_NO_F_HISTORY       = True   # ไม่เคยได้ F (แม้ลงซ้ำแล้วผ่าน)
GOLD_MEDAL_RANK_REQUIRED = True   # ต้องเป็นอันดับ 1 ของหลักสูตร
GOLD_MEDAL_GPAX_MIN      = 3.75   # GPAX ขั้นต่ำสำหรับ Gold Medal
DISMISSAL_CONSECUTIVE     = 2      # Probation ติดต่อกัน N เทอม → Dismissal

# หน่วยกิตรวมตลอดหลักสูตรแต่ละสาขา
CURRICULUM_CREDITS = {
    'เทคโนโลยีปัญญาประดิษฐ์':                  120,
    'เทคโนโลยีสารสนเทศทางธุรกิจ (นานาชาติ)':  126,
    'วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ': 132,
    'เทคโนโลยีสารสนเทศ':                       129,
}

def get_credits_info(curriculum: str, credits_earned: int) -> dict:
    """
    คืนข้อมูลหน่วยกิตของสาขา + คำนวณที่เหลือ
    """
    total = CURRICULUM_CREDITS.get(curriculum)
    if total is None:
        return {"total": None, "earned": credits_earned, "remaining": None, "curriculum": curriculum}
    remaining = max(0, total - credits_earned)
    return {
        "total":      total,
        "earned":     credits_earned,
        "remaining":  remaining,
        "curriculum": curriculum,
        "progress":   round(credits_earned / total * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
# STEP 2 — TRANSCRIPT PARSER
# ═══════════════════════════════════════════════════════════════

@dataclass
class CourseRecord:
    course_id:  str
    name:       str
    credits:    int
    grade:      str
    semester:   str
    is_transfer: bool = False


def normalize_2col_transcript(text: str) -> str:
    """
    Detect และ normalize 2-column PDF transcript → single column
    ไม่เปลี่ยนแปลงถ้าเป็น single column อยู่แล้ว

    2-col format เกิดเมื่อ copy text จาก PDF แล้วข้อมูล 2 คอลัมน์
    มาอยู่ในบรรทัดเดียวกัน เช่น:
      "06046400 CALCULUS 1 3 B+ 90641006 TEAM-PROJECT 3 1"
      "1st Semester, Year, 2023-2024 06046448 Building LLM powered Applications 3"

    วิธีแก้: split แต่ละบรรทัดเป็น (left, right) แล้ว concatenate
    right_lines จะถูก append ต่อท้าย left → parser จะ associate
    กับ semester สุดท้าย (= semester ปัจจุบันที่ล้นไปฝั่งขวา)
    """
    _COURSE_ID  = re.compile(r'\d{8}')
    _SEM_HDR    = re.compile(
        r'\d+(?:st|nd|rd|th)\s+Semester[^\n]*?\d{4}[-\u2013]\d{4}',
        re.IGNORECASE
    )
    # metadata ที่มักอยู่ฝั่งขวา คั่นด้วย ≥2 spaces
    _RIGHT_META = re.compile(
        r'\s{2,}(GPS\s*:.*|Total\s+number.*|Cumulative\s+GPA.*|'
        r'Checked\s+by.*|[-\u2014]+\s*Transcript\s+Closed.*|\([A-Z][a-z].*\))',
        re.IGNORECASE
    )

    lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # ── detect 2-col ──────────────────────────────────────────────
    two_id_count = sum(1 for ln in lines if len(_COURSE_ID.findall(ln)) >= 2)
    sem_id_count = sum(1 for ln in lines if _SEM_HDR.search(ln) and _COURSE_ID.search(ln))
    if two_id_count + sem_id_count < 1:
        return text  # single-column → คืนเลยไม่แก้ไข

    left_lines  = []
    right_lines = []

    for line in lines:
        ids = list(_COURSE_ID.finditer(line))

        if len(ids) >= 2:
            # 2 course IDs บรรทัดเดียว → split ก่อน ID ที่สอง
            split_pos = ids[1].start()
            left_lines.append(line[:split_pos].rstrip())
            right_lines.append(line[split_pos:].strip())

        elif len(ids) == 1 and _SEM_HDR.search(line):
            # semester header (ซ้าย) + course ID (ขวา) บรรทัดเดียวกัน
            split_pos = ids[0].start()
            left_lines.append(line[:split_pos].rstrip())
            right_lines.append(line[split_pos:].strip())

        else:
            # บรรทัดปกติ — ตรวจว่ามี metadata ฝั่งขวาไหม
            # เช่น "06066000 DISCRETE MATH 3 C  GPS : -  GPA : 3.5"
            right_meta = _RIGHT_META.search(line)
            if right_meta:
                left_lines.append(line[:right_meta.start()].rstrip())
                right_lines.append(right_meta.group(1).strip())
            else:
                left_lines.append(line)

    right_content = [r for r in right_lines if r.strip()]
    return '\n'.join(left_lines) + '\n' + '\n'.join(right_content)


def parse_transcript(text: str) -> dict:
    """
    Parse transcript text → structured data
    รองรับทั้ง format ปกติ และ PDF 2 คอลัมน์ที่ layout แตก
    """
    text = normalize_2col_transcript(text)  # normalize ก่อนทุกครั้ง
    courses_raw: dict[str, list[CourseRecord]] = {}  # {course_id: [records]}
    semesters: list[str] = []
    semester_courses: dict[str, list[CourseRecord]] = {}

    current_sem = "Unknown"
    is_transfer_section = False

    lines = text.replace("\r", "\n").split("\n")

    # แยก semester header
    sem_pattern = re.compile(
        r"((?:\d+(?:st|nd|rd|th)\s+Semester.*?\d{4}[-–]\d{4})|"
        r"(?:Transferred\s+Credits?)|"
        r"(?:\d/\d{4}))",
        re.IGNORECASE
    )

    # หา grade lines แบบ standalone (PDF แตก): "3 B+" หรือ "3 F" บรรทัดเดียว
    standalone_grades = re.findall(r"^\s*(\d)\s+(A|B\+|B|C\+|C|D\+|D|F|S|U|W|P)\s*$",
                                   text, re.MULTILINE)

    # หา course lines: รหัส 8 หลัก + ชื่อ + credit + grade (ในบรรทัดเดียว)
    # รองรับทั้งชื่อ uppercase และ mixed case (เช่น "Building LLM powered Applications")
    inline_pattern = re.compile(
        r"(\d{8})\s+([A-Za-z][A-Za-z0-9 ,\-&/()]+?)\s+(\d)\s+(A|B\+|B|C\+|C|D\+|D|F|S|U|W|P)(?:\s|$)"
    )

    # ── ตรวจ semester ที่มี GPS : - (กำลังเรียนอยู่) ───────────────────────────
    # แบ่ง text ออกเป็น semester blocks แล้วตรวจทีละ block
    semester_blocks: list[tuple[str, str]] = []  # [(sem_label, block_text)]
    current_block_lines: list[str] = []
    current_block_sem = "Unknown"

    for line in lines:
        sem_match = sem_pattern.search(line)
        if sem_match:
            if current_block_lines:
                semester_blocks.append((current_block_sem, "\n".join(current_block_lines)))
            sem_text = sem_match.group(1).strip()
            if "transfer" in sem_text.lower():
                is_transfer_section = True
                current_block_sem = "Transferred"
            else:
                is_transfer_section = False
                current_block_sem = sem_text
            current_block_lines = [line]
            if current_block_sem not in semester_courses:
                semester_courses[current_block_sem] = []
                semesters.append(current_block_sem)
        else:
            current_block_lines.append(line)

    if current_block_lines:
        semester_blocks.append((current_block_sem, "\n".join(current_block_lines)))

    # Parse วิชาแต่ละ semester block
    for sem_label, block in semester_blocks:
        for m in inline_pattern.finditer(block):
            cid, name, cred, grade = m.group(1), m.group(2).strip(), int(m.group(3)), m.group(4)
            is_trans = sem_label == "Transferred"
            rec = CourseRecord(cid, name, cred, grade, sem_label, is_trans)
            courses_raw.setdefault(cid, []).append(rec)
            semester_courses.setdefault(sem_label, []).append(rec)

    # ── ตรวจวิชาที่กำลังเรียนอยู่ ─────────────────────────────────────────────
    # วิธีใหม่: หา semester block ที่มี "GPS : -" แล้ว scan หา course ID ที่ไม่มีเกรด
    VALID_GRADES = {"A", "B+", "B", "C+", "C", "D+", "D", "F", "S", "U", "W", "P"}
    # รหัส 8 หลักทั้งหมดที่มีเกรดแล้ว
    graded_ids = set(courses_raw.keys())

    in_progress_courses = []

    for sem_label, block in semester_blocks:
        # เช็คว่าเทอมนี้ยังไม่มีเกรด (GPS : -)
        if not re.search(r"GPS\s*:\s*-", block):
            continue

        # scan หา course ID 8 หลักใน block นี้ที่ไม่มีเกรดตามหลัง
        # process line-by-line: บรรทัดที่ขึ้นด้วย course ID คือ course ใหม่
        # บรรทัดที่ไม่ขึ้นด้วย course ID คือ continuation ชื่อวิชา
        # ต้องตรวจ credit จากบรรทัดแรกของ course ก่อน merge continuation
        block_lines = block.split("\n")
        pending: dict | None = None  # course ที่กำลัง build อยู่

        for ln in block_lines:
            stripped = ln.strip()
            if not stripped:
                continue

            cid_m = re.match(r"^(\d{8})\s+(.*)", stripped)
            if cid_m:
                # บันทึก pending course ที่ค้างไว้ก่อน
                if pending:
                    in_progress_courses.append(pending)
                    pending = None

                cid   = cid_m.group(1)
                rest  = cid_m.group(2).strip()
                tokens = rest.split()
                if not tokens:
                    continue

                last = tokens[-1]
                # มีเกรดท้ายสุด → วิชาที่เรียนผ่านแล้ว ข้ามไป
                if last in VALID_GRADES:
                    continue
                # ไม่มีเกรด ตรวจว่า token สุดท้ายเป็น credit (digit) หรือเปล่า
                if re.match(r"^\d+$", last):
                    pending = {
                        "course_id": cid,
                        "name":      " ".join(tokens[:-1]).strip(),
                        "credits":   int(last),
                    }
            else:
                # continuation line — ต่อท้ายชื่อวิชาใน pending (ถ้ามี)
                # หยุดเมื่อเจอ footer lines
                is_footer = re.match(
                    r"^(?:GPS|GPA|Total|Cumulative|COURSE\s+TITLE)",
                    stripped, re.IGNORECASE
                )
                if pending and not is_footer:
                    pending["name"] = (pending["name"] + " " + stripped).strip()
                elif is_footer and pending:
                    in_progress_courses.append(pending)
                    pending = None

        # บันทึก pending สุดท้าย
        if pending:
            in_progress_courses.append(pending)

    in_progress_credits = sum(c["credits"] for c in in_progress_courses)

    # ── สรุปข้อมูล summary จาก bottom of transcript (แม่นที่สุด) ──────────────
    cum_gpa_match    = re.search(r"Cumulative\s+GPA\s*:\s*(\d+\.\d+)", text, re.IGNORECASE)
    total_cred_match = re.search(r"Total\s+number\s+of\s+credit\s+earned\s*:\s*(\d+)", text, re.IGNORECASE)

    cumulative_gpa    = float(cum_gpa_match.group(1))    if cum_gpa_match    else None
    total_credits     = int(total_cred_match.group(1))   if total_cred_match else None

    # หา GPS/GPA รายเทอมจาก "GPS : 3.50  GPA : 3.50"
    sem_gpa_map: dict[str, float] = {}
    for sgpa_m in re.finditer(r"GPS\s*:\s*(-|\d+\.?\d*)\s+GPA\s*:\s*(\d+\.\d+)", text):
        gpa_val = float(sgpa_m.group(2))
        sem_gpa_map[f"gpa_{len(sem_gpa_map)}"] = gpa_val

    # standalone grade lines (PDF แตก) — เก็บไว้ใน metadata
    standalone_f_count = sum(1 for _, g in standalone_grades if g == "F")
    inline_f_count     = sum(
        1 for records in courses_raw.values()
        for r in records if r.grade == "F"
    )

    return {
        "courses_raw":         courses_raw,
        "semester_courses":    semester_courses,
        "semesters":           semesters,
        "cumulative_gpa":      cumulative_gpa,
        "total_credits":       total_credits,
        "standalone_grades":   standalone_grades,
        "f_count_inline":      inline_f_count,
        "f_count_standalone":  standalone_f_count,
        "in_progress":         in_progress_courses,
        "in_progress_credits": in_progress_credits,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3 — GPA CALCULATOR
# ═══════════════════════════════════════════════════════════════

def calculate_gpa(parsed: dict) -> dict:
    """
    คำนวณ GPA ต่อเทอม + GPAX สะสม
    จัดการ edge cases: F แล้วลงซ้ำ, W, Transfer
    """
    # de-duplicate: วิชาเดียวกัน ใช้เกรดล่าสุด (ยกเว้น W)
    final_courses: dict[str, CourseRecord] = {}
    ever_failed: set[str] = set()

    for cid, records in parsed["courses_raw"].items():
        for r in records:
            if r.grade == "F":
                ever_failed.add(cid)
            if r.grade == "W":
                continue  # ไม่นำมาคำนวณ
        # ใช้ record สุดท้ายที่ไม่ใช่ W
        valid = [r for r in records if r.grade != "W"]
        if valid:
            final_courses[cid] = valid[-1]

    # คำนวณ GPAX จาก final_courses
    total_points  = 0.0
    total_cred_gpa = 0
    transfer_credits = 0
    credits_earned   = 0

    for cid, r in final_courses.items():
        gp = GRADE_POINTS.get(r.grade)
        if gp is None:
            # S/U/P — นับ credits แต่ไม่นับ GPA
            if r.grade == "S":
                credits_earned += r.credits
            if r.is_transfer:
                transfer_credits += r.credits
            continue
        if r.is_transfer:
            transfer_credits += r.credits
        if gp >= 0:  # รวม F ด้วย
            total_points   += gp * r.credits
            total_cred_gpa += r.credits
            if gp > 0:
                credits_earned += r.credits  # F ไม่นับ credits

    calc_gpax = round(total_points / total_cred_gpa, 2) if total_cred_gpa > 0 else 0.0

    # ถ้ามี Cumulative GPA จาก transcript ให้ใช้ค่านั้นแทน (แม่นกว่า)
    gpax = parsed["cumulative_gpa"] if parsed["cumulative_gpa"] is not None else calc_gpax
    credits_earned = parsed["total_credits"] if parsed["total_credits"] is not None else credits_earned

    return {
        "gpax":             gpax,
        "calc_gpax":        calc_gpax,        # ค่าที่คำนวณเอง
        "credits_earned":   credits_earned,
        "transfer_credits": transfer_credits,
        "transfer_ratio":   round(transfer_credits / credits_earned, 2) if credits_earned > 0 else 0,
        "ever_failed":      ever_failed,       # set ของ course_id ที่เคยได้ F
        "final_courses":    final_courses,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 4 — STATUS EVALUATOR
# ═══════════════════════════════════════════════════════════════

def evaluate_status(calc: dict, parsed: dict, consecutive_probation: int = 0,
                    rank_in_program: int = None) -> dict:
    gpax           = calc["gpax"]
    transfer_ratio = calc["transfer_ratio"]
    ever_failed    = calc["ever_failed"]

    # นับ F ทั้งหมด (inline + standalone)
    total_f = len(ever_failed) + parsed.get("f_count_standalone", 0)

    # ── สถานะหลัก ──────────────────────────────────────────────
    if consecutive_probation >= DISMISSAL_CONSECUTIVE:
        status = "เสี่ยง Dismissal"
    elif RULES["probation"](gpax):
        status = "Probation"
    elif RULES["near_probation"](gpax):
        status = "เสี่ยง Probation"
    else:
        status = "ปกติ"

    # ── ตรวจ Honors (เฉพาะสถานะปกติ) ──────────────────────────
    honors = None
    honors_blocked = []

    if status == "ปกติ":
        if HONORS_NO_F_HISTORY and total_f > 0:
            honors_blocked.append(f"เคยได้ F {total_f} วิชา")
        if transfer_ratio > HONORS_MAX_TRANSFER_RATIO:
            honors_blocked.append(f"Transfer credits {transfer_ratio*100:.0f}% เกินเกณฑ์ 30%")

        if not honors_blocked:
            if RULES["gold_medal_track"](gpax):
                if rank_in_program == 1:
                    honors = "First Class Honors with Gold Medal"
                elif rank_in_program is not None:
                    honors = f"Gold Medal Eligible — GPAX ✅ แต่อันดับ {rank_in_program} (ต้องเป็น #1)"
                else:
                    honors = "Gold Medal Eligible (GPAX ≥ 3.75 ✅ | รอยืนยันอันดับ #1 ของหลักสูตร)"
            elif RULES["first_class_honors"](gpax):
                honors = "First Class Honors"
            elif RULES["second_class_honors"](gpax):
                honors = "Second Class Honors"
        else:
            # มี block แต่ GPAX ถึงเกณฑ์ → บอกว่า "เกือบได้"
            if RULES["second_class_honors"](gpax):
                honors = f"GPAX ถึงเกณฑ์ Honors แต่ถูก block: {', '.join(honors_blocked)}"

    return {
        "status":              status,
        "honors":              honors,
        "honors_blocked":      honors_blocked,
        "consecutive_probation": consecutive_probation,
        "total_f_courses":     total_f,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 5 — TARGET GPA CALCULATOR
# ═══════════════════════════════════════════════════════════════

def calculate_target_gpa(
    calc: dict,
    next_credits: int = 18,
    total_credits_remaining: int = None,
) -> dict:
    """
    คำนวณว่าต้องได้ GPA เท่าไหร่เพื่อบรรลุเป้าหมาย
    แสดงผลแยกตามจำนวนเทอมที่เหลือ (1-4 เทอม)
    ถ้าระบุ total_credits_remaining จะใช้หน่วยกิตที่เหลือจริงเป็น cap
    เพื่อให้ตัวเลขสอดคล้องกับหน้า Scenario
    """
    gpax        = calc["gpax"]
    cred_earned = calc["credits_earned"]

    GOALS = {
        "พ้น Probation":       2.00,
        "Second Class Honors": 3.25,
        "First Class Honors":  3.50,
        "Gold Medal Track":    3.75,
    }

    def required_gpa_per_sem(target_gpax: float, n_sems: int) -> Optional[float]:
        """
        สมมติได้ GPA เท่ากันทุกเทอม = x
        (gpax*cred + x*total_new) / (cred + total_new) = target_gpax
        → x = (target_gpax*(cred + total_new) - gpax*cred) / total_new

        ถ้ามี total_credits_remaining ให้ใช้เป็น cap
        เพื่อให้สอดคล้องกับหน่วยกิตที่เหลือจริงในหลักสูตร
        """
        total_new = next_credits * n_sems
        if total_credits_remaining is not None:
            total_new = min(total_new, total_credits_remaining)
        if total_new <= 0:
            return None
        needed = (target_gpax * (cred_earned + total_new) - gpax * cred_earned) / total_new
        return round(needed, 2)

    targets = {}

    for goal_name, goal_gpax in GOALS.items():
        if gpax >= goal_gpax:
            continue  # ถึงเกณฑ์แล้ว ไม่ต้องแสดง

        sem_plans = {}
        for n in range(1, 5):  # 1-4 เทอม
            needed = required_gpa_per_sem(goal_gpax, n)
            if needed is None:
                sem_plans[f"{n} เทอม"] = "ไม่สามารถคำนวณได้"
            elif needed > 4.0:
                sem_plans[f"{n} เทอม"] = f"ต้องได้ 4.00+ (เป็นไปไม่ได้)"
            elif needed < 0:
                sem_plans[f"{n} เทอม"] = "ถึงเกณฑ์แล้ว"
            else:
                sem_plans[f"{n} เทอม"] = needed

        targets[f"{goal_name} (GPAX {goal_gpax})"] = sem_plans

    return {
        "current_gpax":   gpax,
        "credits_so_far": cred_earned,
        "credits_per_sem": next_credits,
        "targets":        targets,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 6 — LLM SUMMARY LAYER
# ═══════════════════════════════════════════════════════════════

def build_llm_prompt(calc: dict, status_result: dict, target: dict) -> str:
    gpax    = calc["gpax"]
    credits = calc["credits_earned"]
    status  = status_result["status"]
    honors  = status_result["honors"]
    targets = target["targets"]
    cps     = target["credits_per_sem"]

    # สร้าง target lines แบบหลายเทอม
    target_lines = ""
    for goal, sem_plans in targets.items():
        target_lines += f"  {goal}:\n"
        for sem, val in sem_plans.items():
            if isinstance(val, float):
                target_lines += f"    - ภายใน {sem} (ลง {cps} หน่วยกิต/เทอม): GPA เฉลี่ย {val:.2f}\n"
            else:
                target_lines += f"    - ภายใน {sem}: {val}\n"

    if not target_lines:
        target_lines = "  - ถึงเกณฑ์เป้าหมายทั้งหมดแล้ว"
        

    prompt = f"""สรุปสถานะการเรียนของนักศึกษาด้วยภาษาที่เป็นมิตรและกระชับ:

ข้อมูล:
- GPAX สะสม: {gpax}
- หน่วยกิตที่ผ่านแล้ว: {credits} หน่วยกิต
- สถานะ: {status}
- เกียรตินิยม: {honors or 'ยังไม่ถึงเกณฑ์'}
- วิชาที่เคยได้ F: {status_result['total_f_courses']} วิชา
- แผน GPA ที่ต้องทำให้ได้เพื่อบรรลุเป้าหมาย (สมมติลง {cps} หน่วยกิต/เทอม):
{target_lines}
{"⚠️ Gold Medal ต้องการ GPAX ≥ 3.75 และเป็นอันดับ 1 ของรุ่นในหลักสูตร" if gpax >= 3.75 else ""}
ให้ตอบ 3-4 ประโยค ขึ้นต้นด้วยสรุปสถานะ ตามด้วยคำแนะนำเชิงปฏิบัติที่เป็นรูปธรรม"""
    return prompt


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def analyze_transcript_full(
    transcript_text: str,
    next_semester_credits: int = 18,
    consecutive_probation: int = 0,
    curriculum: str = None,
) -> dict:
    """
    Entry point หลัก — รับ transcript text คืน structured result
    พร้อม prompt สำหรับส่งต่อให้ LLM
    ถ้าระบุ curriculum จะคำนวณหน่วยกิตที่เหลือให้อัตโนมัติ
    """
    parsed  = parse_transcript(transcript_text)
    calc    = calculate_gpa(parsed)
    status  = evaluate_status(calc, parsed, consecutive_probation)

    # คำนวณหน่วยกิตที่เหลือจาก curriculum
    credits_info = get_credits_info(curriculum or '', calc["credits_earned"])
    remaining    = credits_info["remaining"]

    # ถ้าไม่รู้ remaining ให้ใช้ next_semester_credits ที่ส่งมา
    effective_next = next_semester_credits

    # ส่ง credits_remaining จริงเข้าไปด้วย เพื่อให้ตัวเลข GPA ที่ต้องได้
    # สอดคล้องกับหน้า Scenario ที่ใช้หน่วยกิตจริง
    target  = calculate_target_gpa(calc, effective_next, total_credits_remaining=remaining)
    prompt  = build_llm_prompt(calc, status, target)

    in_progress         = parsed.get("in_progress", [])
    in_progress_credits = parsed.get("in_progress_credits", 0)

    # นับเทอมที่เรียนจบแล้ว (ไม่รวมเทอมปัจจุบันที่ยังไม่มีเกรด)
    all_sems       = parsed.get("semesters", [])
    has_in_progress = len(in_progress) > 0
    # ถ้ามี in_progress → เทอมสุดท้ายใน semesters คือเทอมปัจจุบัน → completed = len-1
    semesters_completed = len(all_sems) - (1 if has_in_progress else 0)

    return {
        "summary": {
            "gpa":                   calc["gpax"],
            "credits_earned":        calc["credits_earned"],
            "credits_total":         credits_info["total"],
            "credits_remaining":     remaining,
            "credits_progress":      credits_info.get("progress"),
            "failed_courses_count":  status["total_f_courses"],
            "in_progress":           in_progress,
            "in_progress_credits":   in_progress_credits,
            "semesters_completed":   semesters_completed,  # เทอมที่เรียนจบแล้ว (ไม่รวมปัจจุบัน)
        },
        "risk": {
            "level":  _risk_level(status),
            "detail": _risk_detail(calc, status, target),
        },
        "honors":     status["honors"],
        "targets":    target["targets"],
        "llm_prompt": prompt,
    }


def _risk_level(status: dict) -> str:
    s = status["status"]
    if "Dismissal" in s or s == "Probation":
        return "วิกฤต"
    elif "เสี่ยง" in s:
        return "เสี่ยง"
    return "ปกติ"


def _risk_detail(calc: dict, status: dict, target: dict) -> list[str]:
    details = []
    gpax = calc["gpax"]

    if status["status"] == "Probation":
        details.append(f"GPAX {gpax:.2f} ต่ำกว่าเกณฑ์ 2.00 อยู่ใน Probation")
    elif status["status"] == "เสี่ยง Probation":
        details.append(f"GPAX {gpax:.2f} ใกล้เกณฑ์ Probation ควรระวัง")
    elif "Dismissal" in status["status"]:
        details.append(f"Probation ติดต่อกัน {status['consecutive_probation']} เทอม เสี่ยงพ้นสภาพ")

    if status["honors_blocked"]:
        details.extend(status["honors_blocked"])

    for goal, sem_plans in target["targets"].items():
        if isinstance(sem_plans, dict):
            v1 = sem_plans.get("1 เทอม")
            if isinstance(v1, float):
                details.append(f"ต้องได้ GPA {v1:.2f} ใน 1 เทอม ({target['credits_per_sem']} หน่วยกิต) เพื่อ {goal}")
            elif isinstance(v1, str) and "ไม่ได้" in v1:
                v2 = sem_plans.get("2 เทอม")
                if isinstance(v2, float):
                    details.append(f"ต้องได้ GPA {v2:.2f} ใน 2 เทอม เพื่อ {goal}")

    return details


# ═══════════════════════════════════════════════════════════════
# STEP 7 — QUICK TEST
# ═══════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════
# SCENARIO SIMULATOR
# ═══════════════════════════════════════════════════════════════

def simulate_scenarios(
    current_gpax: float,
    credits_earned: int,
    remaining_sems: int,
    credits_per_sem: int,
    sem_gpas: list[float],              # GPA ที่สมมติแต่ละเทอม เช่น [4.0, 3.5]
    goals: dict = None,                 # เป้าหมาย เช่น {"First Class": 3.50}
    total_credits_remaining: int = None,  # หน่วยกิตที่เหลือจริงทั้งหมด เช่น 31
) -> dict:
    """
    จำลองแบบ step-by-step ว่าถ้าได้ GPA ตาม sem_gpas
    GPAX จะเป็นเท่าไหร่หลังแต่ละเทอม และต้องทำอีกเท่าไหร่เพื่อบรรลุเป้า

    ถ้าระบุ total_credits_remaining เทอมสุดท้ายจะใช้หน่วยกิตที่เหลือจริง
    เช่น เหลือ 31 หน่วยกิต ลง 18 + 13 แทนที่จะเป็น 18 + 18

    ตัวอย่าง:
        simulate_scenarios(3.67, 89, 2, 18, [4.0], total_credits_remaining=31)
        → เทอม 1: 18 หน่วยกิต → GPAX = 3.70
          เทอม 2: 13 หน่วยกิตที่เหลือจริง → คำนวณแม่นยำขึ้น
    """
    if goals is None:
        goals = {
        "พ้น Probation":                          2.00,
        "Second Class Honors":                    3.25,
        "First Class Honors":                     3.50,
        "Gold Medal Eligible (GPAX ≥ 3.75)":     3.75,  # ← label ชัดขึ้น
        }

    steps = []
    gpax         = current_gpax
    cred         = credits_earned
    credits_used = 0  # หน่วยกิตที่จ่ายไปแล้วในการจำลอง

    for i in range(remaining_sems):
        sem_label = f"เทอมที่ {i+1}"

        # คำนวณหน่วยกิตเทอมนี้ — เทอมสุดท้ายใช้ที่เหลือจริงถ้ามี
        is_last = (i == remaining_sems - 1)
        if is_last and total_credits_remaining is not None:
            actual_cps = max(0, total_credits_remaining - credits_used)
        else:
            actual_cps = credits_per_sem

        new_cred = cred + actual_cps

        if i < len(sem_gpas):
            # เทอมที่สมมติ GPA ไว้แล้ว
            assumed_gpa = sem_gpas[i]
            new_gpax = round(
                (gpax * cred + assumed_gpa * actual_cps) / new_cred, 4
            ) if new_cred > 0 else gpax
            step = {
                "semester":      sem_label,
                "assumed_gpa":   assumed_gpa,
                "gpax_after":    new_gpax,
                "credits_after": new_cred,
                "credits_this_sem": actual_cps,
                "mode":          "assumed",
                "goals_status":  {},
                "goals_needed":  {},
            }

            for gname, gval in goals.items():
                if new_gpax >= gval:
                    step["goals_status"][gname] = "✅ ถึงเกณฑ์แล้ว"
                else:
                    step["goals_status"][gname] = f"ยังขาด {round(gval - new_gpax, 2)}"

            gpax = new_gpax
            cred = new_cred

        else:
            # เทอมที่ยังไม่สมมติ — คำนวณว่าต้องได้เท่าไหร่
            step = {
                "semester":      sem_label,
                "assumed_gpa":   None,
                "gpax_after":    None,
                "credits_after": new_cred,
                "credits_this_sem": actual_cps,
                "mode":          "required",
                "goals_status":  {},
                "goals_needed":  {},
            }

            remaining_after = remaining_sems - i - 1

            for gname, gval in goals.items():
                if gpax >= gval:
                    step["goals_needed"][gname] = "✅ ถึงเกณฑ์แล้ว"
                    continue

                # คำนวณหน่วยกิตที่เหลือในเทอมถัดๆ ไป (รวมเทอมนี้)
                if total_credits_remaining is not None:
                    total_new_cred = max(0, total_credits_remaining - credits_used)
                else:
                    total_new_cred = actual_cps + credits_per_sem * remaining_after

                if total_new_cred <= 0:
                    step["goals_needed"][gname] = "❌ ไม่มีหน่วยกิตเหลือแล้ว"
                    continue

                needed_total = (gval * (cred + total_new_cred) - gpax * cred) / total_new_cred
                needed_total = round(needed_total, 2)

                if needed_total > 4.0:
                    step["goals_needed"][gname] = "❌ ไม่สามารถทำได้ในเวลาที่เหลือ"
                elif needed_total < 0:
                    step["goals_needed"][gname] = "✅ ถึงเกณฑ์แล้ว"
                else:
                    sems_left = remaining_after + 1
                    step["goals_needed"][gname] = f"ต้องได้ GPA เฉลี่ย {needed_total} ใน {sems_left} เทอมที่เหลือ"

            cred = new_cred

        credits_used += actual_cps
        steps.append(step)

    return {
        "initial_gpax":             current_gpax,
        "initial_credits":          credits_earned,
        "remaining_sems":           remaining_sems,
        "credits_per_sem":          credits_per_sem,
        "total_credits_remaining":  total_credits_remaining,
        "assumed_gpas":             sem_gpas,
        "steps":                    steps,
    }


def format_scenario_text(result: dict) -> str:
    """แปลงผล simulate_scenarios เป็นข้อความสำหรับ LLM หรือแสดงผล"""
    lines = [
        f"สถานะเริ่มต้น: GPAX {result['initial_gpax']} | {result['initial_credits']} หน่วยกิต",
        f"สมมติลง {result['credits_per_sem']} หน่วยกิต/เทอม | เหลือ {result['remaining_sems']} เทอม",
        ""
    ]

    for step in result["steps"]:
        lines.append(f"── {step['semester']} ──")
        if step["mode"] == "assumed":
            lines.append(f"  สมมติได้ GPA {step['assumed_gpa']:.2f}")
            lines.append(f"  → GPAX หลังเทอมนี้: {step['gpax_after']:.4f}")
            for gname, gstatus in step["goals_status"].items():
                lines.append(f"  {gname}: {gstatus}")
        else:
            lines.append(f"  เทอมนี้ยังไม่ได้กำหนด GPA")
            for gname, needed in step["goals_needed"].items():
                lines.append(f"  {gname}: {needed}")
        lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    import json

    test_transcript = """
    1st Semester, Year, 2023-2024
    06046400 CALCULUS 1 3 A
    06046402 LINEAR ALGEBRA 3 A
    06066000 DISCRETE MATHEMATICS 3 B
    06066001 PROBABILITY AND STATISTICS 3 B+
    GPS : 3.50  GPA : 3.50

    2nd Semester, Year, 2023-2024
    06046401 CALCULUS 2 3 B+
    06046403 COMPUTER PROGRAMMING 3 A
    06046404 FUNDAMENTAL OF EMBEDDED SYSTEM 3 A
    GPS : 3.90  GPA : 3.70

    Total number of credit earned: 89
    Cumulative GPA: 3.67
    """

    result = analyze_transcript_full(test_transcript, next_semester_credits=18)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n── LLM Prompt ──")
    print(result["llm_prompt"])