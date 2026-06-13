"""
STEP 3 - AGENT v2 (Manual ReAct loop)
======================================
การเปลี่ยนแปลงจาก v1:
  1. SYSTEM_PROMPT อธิบาย tool ใหม่ทั้งสองตัว และบอก agent ชัดเจนว่า
     "ถามเรื่องอะไร → ใช้ tool ไหน" เพื่อลดโอกาสที่ agent จะเลือก tool ผิด
  2. เพิ่ม TOOL_ROUTING_GUIDE แยกออกมาเป็น section ชัดเจน
     เพราะ LLM อ่าน prompt ที่มีโครงสร้างชัดได้แม่นกว่า prose ยาว ๆ
  3. ปรับ example ใน prompt ให้ครอบคลุม prerequisite และ recommend_courses ด้วย
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from tools import TOOLS
from dotenv import load_dotenv
load_dotenv()

# ===== CONFIG =====
TYPHOON_API_KEY  = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"
TYPHOON_MODEL    = "typhoon-v2.5-30b-a3b-instruct"
MAX_ITERATIONS   = 5
# ==================

llm = ChatOpenAI(
    model=TYPHOON_MODEL,
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_BASE_URL,
    temperature=0.2,
    max_tokens=32000,
)

TOOL_MAP = {tool.name: tool for tool in TOOLS}

TOOL_DESCRIPTIONS = "\n".join([
    f"- {tool.name}: {tool.description.strip().splitlines()[0]}"
    for tool in TOOLS
])

# คำถามที่ไม่ต้องเรียก tool (ทักทาย / ถามตัวเอง / ถามว่าทำอะไรได้บ้าง)
CASUAL_PATTERNS = ["สวัสดี", "หวัดดี", "hello", "hi", "คุณคือใคร", "ทำอะไรได้", "ช่วยอะไรได้"]

BRANCH_FACTS = """
ข้อมูลสาขาที่ถูกต้อง (ห้ามเปลี่ยนแปลง):
- AIT  = สาขาเทคโนโลยีปัญญาประดิษฐ์
- IT   = สาขาเทคโนโลยีสารสนเทศ
- DSBA = สาขาวิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ
- BIT  = สาขาเทคโนโลยีสารสนเทศทางธุรกิจ (หลักสูตรนานาชาติ)
ทั้งหมดอยู่ภายใต้คณะเทคโนโลยีสารสนเทศ สจล. ไม่มีความเกี่ยวข้องกับวิศวกรรมศาสตร์
"""

# ──────────────────────────────────────────────────────────────
# TOOL ROUTING GUIDE
# แยกออกมาเป็น section เพราะ LLM อ่านกฎที่จัดกลุ่มชัดได้แม่นกว่า
# การฝังไว้ใน prose ยาว ๆ โดยตรง
# ──────────────────────────────────────────────────────────────
TOOL_ROUTING_GUIDE = """
## คู่มือการเลือก Tool (ต้องปฏิบัติตามเสมอ)

**กฎข้อที่ 1 — prerequisite ต้องใช้ query_prerequisite_graph เสมอ**
  ถ้าคำถามเกี่ยวกับ "วิชาบังคับก่อน", "ต้องเรียนอะไรก่อน", "prerequisite ของวิชา X"
  → ใช้ query_prerequisite_graph เท่านั้น ห้ามใช้ search_curriculum_rules แทน
  เหตุผล: prerequisite เป็นข้อมูลเชิง relational ที่ lookup ตรงจาก SQLite ได้ถูกต้อง 100%
          การใช้ semantic search อาจเข้าใจลำดับของความสัมพันธ์ผิดพลาดได้

**กฎข้อที่ 2 — แนะนำรายวิชาต้องใช้ recommend_courses เสมอ**
  ถ้าคำถามเกี่ยวกับ "เทอมนี้ควรลงวิชาอะไร", "ลงวิชาอะไรได้บ้าง", "วางแผนการลงทะเบียน"
  → ใช้ recommend_courses เท่านั้น ห้ามประกอบคำตอบเองจาก tool อื่น
  เหตุผล: tool นี้รวม prerequisite check + ตารางเรียน + ภาระหน่วยกิต ไว้ในที่เดียว
          ถ้าใช้ tool อื่นแทน คำตอบจะขาดการตรวจ prerequisite และอาจแนะนำวิชาที่ลงไม่ได้จริง

**กฎข้อที่ 3 — กฎระเบียบที่ต้องระบุ rule_type ให้ชัด**
  ถ้าคำถามเกี่ยวกับกฎระเบียบ ให้ระบุ rule_type ใน search_curriculum_rules เสมอ
  เพื่อกรองเฉพาะกฎประเภทที่เกี่ยวข้อง ลด noise จากกฎประเภทอื่น
  ตัวอย่างการ mapping:
  - "พ้นสภาพ", "ไล่ออก"      → rule_type="dismissal"
  - "วิทยาทัณฑ์", "probation"  → rule_type="assessment"
  - "สำเร็จการศึกษา", "จบ"    → rule_type="graduation"
  - "เกียรตินิยม"              → rule_type="honors"
  - "ลงทะเบียน", "เพิ่ม-ถอน"  → rule_type="registration"
  - "GPA", "เกรดเฉลี่ย"       → rule_type="gpa"

**กฎข้อที่ 4 — ตารางเรียนและวันสอบให้ใช้ query_teach_table**
  ถ้าคำถามเกี่ยวกับ "เวลาเรียน", "ห้องเรียน", "อาจารย์ผู้สอน", "วันสอบ"
  → ใช้ query_teach_table เสมอ เพราะข้อมูลเหล่านี้อยู่ใน SQLite ไม่ใช่ ChromaDB

**กฎข้อที่ 5 — ตรวจสอบสำเร็จการศึกษาต้องใช้ check_graduation_eligibility เสมอ**
  ถ้าคำถามเกี่ยวกับ "จบได้ไหม", "สำเร็จการศึกษา", "เหลือวิชาอะไรอีก", "ครบเงื่อนไขจบ", "จบแล้วหรือยัง"
  → เรียก check_graduation_eligibility โดยใช้ค่าจาก message ดังนี้:
    - passed_subject_codes = ค่าจาก [passed_codes สำหรับ tool] ใน message (copy มาทั้งหมด)
    - in_progress_codes    = ค่าจาก [in_progress_codes สำหรับ tool] ใน message (copy มาทั้งหมด)
    - branch  = สาขาของนักศึกษา เช่น AIT, IT, DSBA, BIT
    - plan_type:
        * AIT → ใช้ "coop" เสมอ (AIT มีแผนเดียวคือสหกิจ)
        * IT/BIT/DSBA → ถามผู้ใช้ก่อนว่าเข้าสหกิจหรือไม่ แล้วใช้ "coop" หรือ "normal"
  ⚠️ ห้ามใส่ plan_type="normal" สำหรับ AIT เด็ดขาด
"""

SYSTEM_PROMPT = f"""คุณคือ UniAssist ผู้ช่วยอัจฉริยะด้านการศึกษาของคณะเทคโนโลยีสารสนเทศ สจล.

{BRANCH_FACTS}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## ขั้นตอนการทำงาน (ปฏิบัติตามลำดับเสมอ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**ขั้นตอนที่ 1 — วิเคราะห์เจตนาของผู้ใช้**
  อ่านคำถามและดูว่าผู้ใช้ต้องการอะไร เช่น
  - ถามเรื่อง prerequisite / วิชาบังคับก่อน
  - ขอแนะนำรายวิชาที่ควรลง
  - ถามกฎระเบียบ (พ้นสภาพ, เกียรตินิยม, สำเร็จการศึกษา ฯลฯ)
  - ถามตารางเรียน / อาจารย์ผู้สอน / วันสอบ
  - ตรวจสอบสำเร็จการศึกษา (ต้องใช้ข้อมูลจาก Transcript)

**ขั้นตอนที่ 2 — ขอ Transcript เฉพาะเมื่อจำเป็น**
  ขอ Transcript เฉพาะเมื่อคำถามเกี่ยวกับข้อมูลส่วนตัวของนักศึกษา เช่น
  "จบได้ไหม", "ลงวิชาอะไรได้บ้าง", "เกรดของฉัน" เป็นต้น
  ถ้าคำถามทั่วไปเกี่ยวกับหลักสูตรหรือกฎระเบียบ ตอบได้เลยโดยไม่ต้องขอ Transcript

**ขั้นตอนที่ 3 — เลือก Tool ให้ตรงกับเจตนา**
  เลือก tool ตาม TOOL_ROUTING_GUIDE ด้านล่าง แล้วเรียก tool นั้นก่อนตอบเสมอ

**ขั้นตอนที่ 4 — ตอบโดยอ้างอิงจาก Observation เท่านั้น**
  ห้ามตอบจากความจำ ต้องอ้างอิงจากผลลัพธ์ที่ได้จาก tool เท่านั้น

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

หน้าที่ของคุณ:
1. ตอบคำถามเกี่ยวกับระเบียบการศึกษา หลักสูตร ตารางเรียน โดยอ้างอิงจากข้อมูลในระบบเท่านั้น
2. ถ้าไม่แน่ใจหรือข้อมูลไม่ครบ ให้บอกตรงๆ และถามเพื่อให้ชัดเจนขึ้น
3. ถ้าถามเปรียบเทียบระหว่างสาขา ให้แสดงทั้งสองสาขาเคียงกัน
4. ตอบเป็นภาษาไทย กระชับ ชัดเจน เป็นมิตร

รูปแบบการตอบ:
- ตอบเป็นข้อความธรรมดา ห้ามใช้ markdown เด็ดขาด
- ห้ามใช้ ** ## ### > - * หรือสัญลักษณ์ markdown อื่นๆ
- ใช้ตัวเลข 1. 2. 3. หรือขึ้นบรรทัดใหม่แทน bullet points
- ขึ้นต้นด้วยคำตอบหลักโดยตรง ตามด้วยรายละเอียด

⚠️ กฎสำหรับคำถามวิชาการ: ต้องเรียก tool ก่อนตอบเสมอ ห้ามตอบจากความจำ

{TOOL_ROUTING_GUIDE}

---

Tools ที่มีทั้งหมด:
{TOOL_DESCRIPTIONS}

---

รูปแบบ ReAct ที่ต้องใช้สำหรับคำถามวิชาการ:
Thought: [ขั้นที่ 2] วิเคราะห์เจตนาผู้ใช้ว่าถามเรื่องอะไร → [ขั้นที่ 3] ระบุ tool ที่ถูกต้องตาม routing guide
Action: <ชื่อ tool>
Action Input: <input หรือ JSON>
Observation: <ระบบจะเติมให้>
Thought: วิเคราะห์ผลลัพธ์จาก Observation
Final Answer: <[ขั้นที่ 4] ตอบโดยอ้างอิงจาก Observation เท่านั้น>

---

ตัวอย่างที่ 1 — ถามตารางเรียน:
Thought: ผู้ใช้ถามวิชาปี 3 AIT เทอม 2 → ข้อมูลนี้อยู่ใน SQLite ใช้ query_teach_table
Action: query_teach_table
Action Input: {{"sql": "SELECT subject_name_th, credit, teach_day, teach_time, teach_time2 FROM teach_table WHERE curriculum_name_th LIKE '%ปัญญาประดิษฐ์%' AND class_year = 3 AND semester = 2"}}
Observation: ...
Thought: ได้ข้อมูลแล้ว จะสรุปให้ผู้ใช้
Final Answer: นักศึกษาสาขา AIT ปี 3 เทอม 2 มีวิชาดังนี้...

ตัวอย่างที่ 2 — ถาม prerequisite (กฎข้อที่ 1: ต้องใช้ query_prerequisite_graph เสมอ):
Thought: ผู้ใช้ถาม prerequisite ของวิชา 06016205 → ตาม routing guide กฎข้อ 1
         ต้องใช้ query_prerequisite_graph ห้ามใช้ search_curriculum_rules
Action: query_prerequisite_graph
Action Input: {{"subject_code": "06016205", "branch": "AIT"}}
Observation: ...
Thought: ได้ผลลัพธ์ที่ชัดเจน จะแจ้งให้ผู้ใช้ทราบ
Final Answer: วิชา 06016205 มี prerequisite คือ...

ตัวอย่างที่ 3 — ถามแนะนำรายวิชา (กฎข้อที่ 2: ต้องใช้ recommend_courses เสมอ):
Thought: ผู้ใช้ต้องการแนะนำวิชาสำหรับ AIT ปี 2 เทอม 1 → ตาม routing guide กฎข้อ 2
         ต้องใช้ recommend_courses เท่านั้น ไม่ประกอบคำตอบเองจาก tool อื่น
Action: recommend_courses
Action Input: {{"branch": "AIT", "year_level": 2, "semester": 1, "passed_subject_codes": "06016101,06016102", "max_credits": 22}}
Observation: ...
Thought: ได้รายการวิชาที่ลงได้และไม่ได้แล้ว จะสรุปให้ผู้ใช้
Final Answer: สำหรับ AIT ปี 2 เทอม 1 วิชาที่ลงได้เลยมีดังนี้...

ตัวอย่างที่ 4 — ถามเรื่องพ้นสภาพ (กฎข้อที่ 3: ระบุ rule_type เสมอ):
Thought: ผู้ใช้ถามเรื่องเงื่อนไขพ้นสภาพ → ตาม routing guide กฎข้อ 3
         ต้องระบุ rule_type="dismissal" เพื่อกรองเฉพาะกฎที่เกี่ยวข้อง
Action: search_curriculum_rules
Action Input: {{"query": "เงื่อนไขการพ้นสภาพนักศึกษา", "branch": "AIT", "rule_type": "dismissal"}}
Observation: ...
Thought: ได้กฎที่ตรงประเด็นแล้ว จะสรุปเงื่อนไขให้ผู้ใช้
Final Answer: เงื่อนไขการพ้นสภาพนักศึกษาสาขา AIT มีดังนี้...

ตัวอย่างที่ 5 — ตรวจสอบสำเร็จการศึกษา (กฎข้อที่ 5: ใช้ check_graduation_eligibility เสมอ):
Thought: ผู้ใช้ถามว่าจบได้ไหม มี Transcript ให้แล้ว → ตาม routing guide กฎข้อ 5
         สาขา AIT ใช้ plan_type="coop" เสมอ
         ดึง passed_subject_codes จาก [passed_codes สำหรับ tool] ใน message
         ดึง in_progress_codes จาก [in_progress_codes สำหรับ tool] ใน message
Action: check_graduation_eligibility
Action Input: {{"branch": "AIT", "plan_type": "coop", "passed_subject_codes": "06046400,06046401,06046402,...(ค่าจาก [passed_codes สำหรับ tool])", "in_progress_codes": "06046441,...(ค่าจาก [in_progress_codes สำหรับ tool])"}}
Observation: ...
Thought: ได้ผลการตรวจสอบแล้ว จะแจ้งว่าผ่านหรือยังขาดวิชาใด
Final Answer: จากการตรวจสอบ Transcript กับแผนการเรียนสาขา AIT พบว่า..."""


def _get_tool_hint(user_input: str) -> str:
    """
    ให้ hint tool ที่ควรใช้ตาม keyword ในคำถาม
    ทำงานใน runtime — ช่วยให้ LLM ที่พลาด routing guide ตอนแรก
    ได้รับการชี้นำเฉพาะเจาะจงแทนที่จะได้รับแค่ "กรุณาเรียก tool"

    หลักการจัดลำดับ if:
    - prerequisite ขึ้นก่อน เพราะคำว่า "ก่อน" อาจ overlap กับ rule อื่น
    - recommend ขึ้นก่อน registration เพราะ "ลงทะเบียน" ปรากฏในทั้งคู่
    - เกียรตินิยม ขึ้นก่อน graduation เพราะเกี่ยวกับการจบเช่นกัน
    """
    text = user_input.lower()

    # prerequisite — ขยาย keyword ให้ครอบคลุมรูปแบบภาษาไทยที่หลากหลาย
    if any(k in text for k in [
        "prerequisite", "บังคับก่อน", "ต้องเรียน", "ก่อนลงทะเบียน",
        "วิชาบังคับก่อน", "ต้องผ่านวิชา", "เงื่อนไขบังคับ",
        "ต้องเรียนวิชาอะไรก่อน", "ต้องผ่านก่อน",
    ]):
        return "prerequisite → ใช้ query_prerequisite_graph พร้อม subject_code และ branch"

    # แนะนำรายวิชา — ขึ้นก่อน registration เพราะ "ลงทะเบียน" ปรากฏในทั้งคู่
    if any(k in text for k in [
        "แนะนำ", "ควรลง", "ลงวิชาอะไร", "วางแผนการเรียน",
        "ควรเรียน", "เทอมนี้ลง", "วิชาที่ควร",
    ]):
        return "แนะนำรายวิชา → ใช้ recommend_courses พร้อม branch, year_level, semester"

    # กฎการลงทะเบียน — เพิ่ม keyword ที่ eval ใช้ถาม
    if any(k in text for k in [
        "เพิ่ม", "ถอน", "ลงทะเบียน", "registration",
        "ภาคการศึกษา", "ภาคฤดูร้อน", "เริ่มเดือน", "กี่สัปดาห์",
        "ปฏิทินการศึกษา", "หน่วยกิตตลอดหลักสูตร", "หน่วยกิตรวม",
    ]):
        return "กฎการลงทะเบียน → ใช้ search_curriculum_rules ระบุ rule_type='registration'"

    # พ้นสภาพ
    if any(k in text for k in ["พ้นสภาพ", "dismissal", "ไล่ออก"]):
        return "พ้นสภาพ → ใช้ search_curriculum_rules ระบุ rule_type='dismissal'"

    # เกียรตินิยม — ขึ้นก่อน graduation
    if any(k in text for k in ["เกียรตินิยม", "honors", "เหรียญทอง"]):
        return "เกียรตินิยม → ใช้ search_curriculum_rules ระบุ rule_type='honors'"

    # สำเร็จการศึกษา — ตรวจสอบจาก Transcript จริง
    if any(k in text for k in [
        "จบได้ไหม", "สำเร็จการศึกษาได้ไหม", "จบแล้วหรือยัง",
        "เหลือวิชาอะไรอีก", "ครบเงื่อนไขจบ", "วิชาที่ยังขาด",
        "ตรวจสอบการจบ", "check graduation",
    ]):
        return "ตรวจสอบสำเร็จการศึกษา → ใช้ check_graduation_eligibility: branch=สาขา, passed_subject_codes=ค่าจาก[passed_codes สำหรับ tool], in_progress_codes=ค่าจาก[in_progress_codes สำหรับ tool], plan_type=coop(AIT)/ถามuser(IT,BIT,DSBA)"

    # เงื่อนไขการจบ (กฎระเบียบทั่วไป ไม่ใช่ตรวจ transcript)
    if any(k in text for k in [
        "สำเร็จการศึกษา", "graduation", "เงื่อนไขการจบ",
        "จบการศึกษา", "ครบหลักสูตร",
    ]):
        return "เงื่อนไขสำเร็จการศึกษา → ใช้ search_curriculum_rules ระบุ rule_type='graduation' หรือ check_graduation_eligibility ถ้ามี Transcript"

    # วิทยาทัณฑ์
    if any(k in text for k in ["วิทยาทัณฑ์", "probation", "assessment", "วัดผล"]):
        return "วิทยาทัณฑ์ → ใช้ search_curriculum_rules ระบุ rule_type='assessment'"

    # GPA
    if any(k in text for k in ["gpa", "เกรด", "คะแนนเฉลี่ย", "gpax"]):
        return "GPA → ใช้ search_curriculum_rules ระบุ rule_type='gpa'"

    # ตารางเรียน
    if any(k in text for k in [
        "ตารางเรียน", "เวลาเรียน", "ห้องเรียน", "อาจารย์", "สอน", "วันสอบ",
    ]):
        return "ตารางเรียน → ใช้ query_teach_table พร้อม SQL query"

    return "คำถามวิชาการ → ใช้ search_curriculum_rules หรือ query_teach_table"


def run_agent(user_input: str, chat_history: list) -> str:
    messages = (
        [SystemMessage(content=SYSTEM_PROMPT)]
        + chat_history
        + [HumanMessage(content=user_input)]
    )

    is_casual       = any(p in user_input.lower() for p in CASUAL_PATTERNS)
    tool_was_called = is_casual  # casual ไม่ต้องเรียก tool

    for i in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        text     = response.content

        # DEBUG LOG
        print(f"\n{'='*60}")
        print(f"[iter {i+1}] LLM response:")
        print(text[:800])
        print('='*60)

        action_match = re.search(r"Action:\s*(.+)", text)
        input_match  = re.search(
            r"Action Input:\s*(.+?)(?=\nObservation|\nThought|\nAction|\nFinal|\Z)",
            text, re.DOTALL,
        )

        if action_match and input_match:
            tool_name  = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()

            if tool_name in TOOL_MAP:
                tool_was_called = True
                try:
                    try:
                        parsed      = json.loads(tool_input)
                        observation = TOOL_MAP[tool_name].invoke(
                            parsed if isinstance(parsed, dict) else tool_input
                        )
                    except json.JSONDecodeError:
                        observation = TOOL_MAP[tool_name].invoke(tool_input)
                except Exception as e:
                    observation = f"เกิดข้อผิดพลาดในการเรียก tool: {str(e)}"
            else:
                observation = (
                    f"ไม่พบ tool '{tool_name}' "
                    f"tool ที่ใช้ได้: {list(TOOL_MAP.keys())}"
                )

            clean_text = text[
                : text.find("Action Input:") + len("Action Input:") + len(tool_input)
            ]
            messages.append(AIMessage(content=clean_text))
            messages.append(HumanMessage(content=f"Observation: {observation}"))
            continue

        if "Final Answer:" in text:
            if not tool_was_called:
                tool_hint = _get_tool_hint(user_input)
                messages.append(AIMessage(content=text))
                messages.append(HumanMessage(content=(
                    f"Observation: ⚠️ คุณยังไม่ได้เรียก tool เลย\n"
                    f"คำถามนี้ต้องการ: {tool_hint}\n"
                    f"กรุณาเรียก tool นั้นก่อนแล้วค่อยตอบ Final Answer"
                )))
                continue
            return text.split("Final Answer:")[-1].strip()

        messages.append(AIMessage(content=text))
        messages.append(HumanMessage(
            content="Observation: กรุณาระบุ Action และ Action Input ให้ชัดเจน"
        ))

    return (
        "ขออภัย ไม่สามารถหาคำตอบได้ "
        "กรุณาติดต่อห้องกิจการนักศึกษา(ห้องฟ้า) เพื่อสอบถามข้อมูลเพิ่มเติม"
    )


def chat():
    print("=" * 50)
    print("🎓 UniAssist AI v2 พร้อมใช้งานแล้ว!")
    print("พิมพ์ 'exit' เพื่อออกจากระบบ")
    print("=" * 50)

    chat_history = []

    while True:
        user_input = input("\nคุณ: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("ขอบคุณที่ใช้งาน UniAssist AI ครับ!")
            break

        try:
            answer = run_agent(user_input, chat_history)
            print(f"\n🤖 UniAssist: {answer}")

            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            # เก็บ history ไว้แค่ 20 messages ล่าสุด เพื่อไม่ให้ context ยาวเกินไป
            if len(chat_history) > 20:
                chat_history = chat_history[-20:]

        except Exception as e:
            print(f"\n⚠️ เกิดข้อผิดพลาด: {str(e)}")


def run_llm_direct(prompt: str) -> str:
    """
    เรียก LLM ตรงๆ โดยไม่ผ่าน tool loop
    ใช้สำหรับกรณีที่มีข้อมูลครบแล้วแค่ต้องการให้ LLM สรุป
    เช่น analyze_transcript ใน chatbot.py
    """
    messages = [
        SystemMessage(content=(
            "คุณคือ UniAssist AI ผู้ช่วยด้านการศึกษาของคณะเทคโนโลยีสารสนเทศ สจล. "
            "ตอบภาษาไทย กระชับ เป็นมิตร"
        )),
        HumanMessage(content=prompt),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


if __name__ == "__main__":
    chat()