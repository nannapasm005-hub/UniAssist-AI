"""
extract_plan.py  (OCR version — ใช้ pytesseract ฟรี 100%)
==========================================================
แปลงแผนการศึกษา PDF → CSV โดย:
  1. pdftoppm  : rasterize PDF → JPEG
  2. pytesseract: OCR ภาษาไทย+อังกฤษ
  3. Typhoon   : parse text → structured JSON (text-only ไม่ต้องการ vision)
"""

import base64, csv, json, re, subprocess, tempfile, os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import pytesseract
from PIL import Image

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
PDF_DIR       = Path(".")
OUT_DIR       = Path(".")
DPI           = 200                              # สูงขึ้นเพื่อ OCR ที่ดีขึ้น
TYPHOON_MODEL = "typhoon-v2.5-30b-a3b-instruct"
TYPHOON_URL   = "https://api.opentyphoon.ai/v1"
TESS_LANG     = "tha+eng"                        # ภาษาไทย + อังกฤษ
# ─────────────────────────────────────────────────────────────────────────────

BRANCH_MAP = {
    "AIT":  "เทคโนโลยีปัญญาประดิษฐ์",
    "BIT":  "เทคโนโลยีสารสนเทศทางธุรกิจ (นานาชาติ)",
    "DSBA": "วิทยาการข้อมูลและการวิเคราะห์เชิงธุรกิจ",
    "IT":   "เทคโนโลยีสารสนเทศ",
}

PARSE_PROMPT = """ข้อความด้านล่างนี้มาจาก OCR ของหน้าแผนการศึกษาสาขา {branch} ของมหาวิทยาลัย

{ocr_text}

กรุณา extract รายวิชาทั้งหมดในหน้านี้ออกมาเป็น JSON array ตามรูปแบบนี้:
[
  {
    "branch": "{branch}",
    "year": 1,
    "semester": 1,
    "subject_code": "06046400",
    "subject_name_th": "แคลคูลัส 1",
    "subject_name_en": "CALCULUS 1",
    "credits": 3,
    "credit_detail": "3-0-6"
  }
]

กฎ:
- year และ semester ให้อ่านจากหัวตาราง เช่น "ปีที่ 1 ภาคการศึกษาที่ 2"
- ถ้า subject_code ไม่ชัดหรือเป็น xx ให้ใส่ "ELECTIVE"
- ถ้าไม่มีข้อมูลรายวิชาในหน้านี้ให้ตอบ []
- ตอบเป็น JSON array เท่านั้น ห้ามมีข้อความอื่น"""


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list:
    out_dir = Path(tempfile.mkdtemp(prefix="pdf_pages_"))
    prefix  = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True, capture_output=True
    )
    return sorted(out_dir.glob("page-*.jpg"))


def ocr_image(img_path: Path) -> str:
    """OCR หน้าหนึ่งด้วย tesseract"""
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img, lang=TESS_LANG)
    return text.strip()


def parse_with_typhoon(client: OpenAI, ocr_text: str, branch: str) -> list:
    """ส่ง OCR text ให้ Typhoon แปลงเป็น structured data"""
    if len(ocr_text) < 30:
        return []

    prompt = PARSE_PROMPT.replace("{branch}", branch).replace("{ocr_text}", ocr_text)

    response = client.chat.completions.create(
        model=TYPHOON_MODEL,
        max_tokens=4096,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for row in data:
                row["branch"] = branch
            return data
    except json.JSONDecodeError:
        print(f"  warning: parse JSON ไม่ได้: {raw[:100]}...")
    return []


def process_pdf(client: OpenAI, pdf_path: Path, branch: str) -> list:
    print(f"\nกำลังประมวลผล {pdf_path.name} (สาขา {branch})")
    print("  Rasterizing...")
    images = pdf_to_images(pdf_path, DPI)
    print(f"  พบ {len(images)} หน้า")

    all_subjects = []
    for i, img in enumerate(images, 1):
        print(f"  หน้า {i}/{len(images)}: OCR...", end=" ", flush=True)
        ocr_text = ocr_image(img)
        print(f"Typhoon parse...", end=" ", flush=True)
        subjects = parse_with_typhoon(client, ocr_text, branch)
        print(f"พบ {len(subjects)} วิชา")
        all_subjects.extend(subjects)
        img.unlink(missing_ok=True)

    return all_subjects


def save_csv(subjects: list, out_path: Path):
    if not subjects:
        print(f"  ไม่มีข้อมูลสำหรับ {out_path.name}")
        return
    fieldnames = ["branch", "year", "semester", "subject_code",
                  "subject_name_th", "subject_name_en", "credits", "credit_detail"]
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(subjects)
    print(f"  บันทึก {len(subjects)} แถว -> {out_path}")


def main():
    api_key = os.getenv("TYPHOON_API_KEY")
    if not api_key:
        raise ValueError("ไม่พบ TYPHOON_API_KEY ใน .env")

    client = OpenAI(api_key=api_key, base_url=TYPHOON_URL)
    all_subjects = []

    for branch in BRANCH_MAP:
        pdf_path = PDF_DIR / f"แผน_{branch}.pdf"
        if not pdf_path.exists():
            print(f"ไม่พบไฟล์ {pdf_path} — ข้ามไป")
            continue
        subjects = process_pdf(client, pdf_path, branch)
        all_subjects.extend(subjects)
        save_csv(subjects, OUT_DIR / f"study_plan_{branch}.csv")

    if all_subjects:
        save_csv(all_subjects, OUT_DIR / "study_plan_all.csv")
        print(f"\nเสร็จแล้ว! รวม {len(all_subjects)} วิชาจากทุกสาขา")
    else:
        print("\nไม่พบข้อมูลเลย")


if __name__ == "__main__":
    main()