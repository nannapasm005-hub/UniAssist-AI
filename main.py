from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import jwt, JWTError
from pydantic import BaseModel
from typing import Optional
import bcrypt
import sqlite3, datetime
from dotenv import load_dotenv
load_dotenv()

from chatbot import chat as chatbot_chat, analyze_transcript

app = FastAPI()
templates = Jinja2Templates(directory="templates")

SECRET_KEY = "uniassist-secret-key"
ALGORITHM  = "HS256"

# ── DB ────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect("uniassist.db")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("ALTER TABLE STUDENTS ADD COLUMN password TEXT")
    conn.commit()
    conn.close()

try:
    init_db()
except:
    pass  # คอลัมน์มีอยู่แล้ว

# ── Auth helpers ──────────────────────────────────────────────────────────────
def create_token(student_id: str):
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return jwt.encode({"sub": student_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request):
    token = request.cookies.get("token")
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    student_id = get_current_user(request)
    if not student_id:
        return RedirectResponse("/login")
    conn = get_db()
    student = conn.execute("SELECT * FROM STUDENTS WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()
    if not student:
        return RedirectResponse("/login")
    return templates.TemplateResponse(request=request, name="home.html", context={"student": dict(student)})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={"error": None})

@app.post("/login", response_class=HTMLResponse)
def login(request: Request, student_id: str = Form(...), password: str = Form(...)):
    conn = get_db()
    student = conn.execute("SELECT * FROM STUDENTS WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()

    if not student:
        return templates.TemplateResponse(request=request, name="login.html", context={"error": "ไม่พบรหัสนักศึกษานี้ในระบบ"})

    if not student["password"]:
        return RedirectResponse(f"/setup-password?student_id={student_id}", status_code=303)

    if not bcrypt.checkpw(password.encode("utf-8")[:72], student["password"].encode("utf-8")):
        return templates.TemplateResponse(request=request, name="login.html", context={"error": "รหัสผ่านไม่ถูกต้อง"})

    token = create_token(student_id)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("token", token, httponly=True)
    return response

@app.get("/setup-password", response_class=HTMLResponse)
def setup_page(request: Request, student_id: str):
    return templates.TemplateResponse(request=request, name="setup_password.html", context={"student_id": student_id, "error": None})

@app.post("/setup-password", response_class=HTMLResponse)
def setup_password(request: Request, student_id: str = Form(...), password: str = Form(...), confirm: str = Form(...)):
    if password != confirm:
        return templates.TemplateResponse(request=request, name="setup_password.html", context={"student_id": student_id, "error": "รหัสผ่านไม่ตรงกัน"})
    if len(password) < 6:
        return templates.TemplateResponse(request=request, name="setup_password.html", context={"student_id": student_id, "error": "รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร"})

    hashed = bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")
    conn = get_db()
    conn.execute("UPDATE STUDENTS SET password = ? WHERE student_id = ?", (hashed, student_id))
    conn.commit()
    conn.close()

    token = create_token(student_id)
    response = RedirectResponse("/", status_code=303)
    response.set_cookie("token", token, httponly=True)
    return response

@app.get("/logout")
def logout():
    response = RedirectResponse("/login")
    response.delete_cookie("token")
    return response

# ── Chatbot API ───────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    transcript_text: Optional[str] = None
    chat_history: Optional[list] = []

class TranscriptRequest(BaseModel):
    transcript_text: str

@app.post("/api/chat")
def api_chat(body: ChatRequest, request: Request):
    student_id = get_current_user(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบก่อน")

    result = chatbot_chat(
        message=body.message,
        transcript_text=body.transcript_text,
        chat_history=body.chat_history or [],
    )
    return result

@app.post("/api/analyze-transcript")
def api_analyze_transcript(body: TranscriptRequest, request: Request):
    student_id = get_current_user(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบก่อน")

    result = analyze_transcript(body.transcript_text, student_id=student_id)
    return result

# ── Scenario Simulator API ────────────────────────────────────────────────────
class ScenarioRequest(BaseModel):
    current_gpax: float
    credits_earned: int
    remaining_sems: int
    credits_per_sem: int
    sem_gpas: list              # GPA ที่สมมติแต่ละเทอม เช่น [4.0, 3.5]
    credits_remaining: Optional[int] = None  # หน่วยกิตที่เหลือจริงทั้งหมด เช่น 31

@app.post("/api/simulate")
def api_simulate(body: ScenarioRequest, request: Request):
    student_id = get_current_user(request)
    if not student_id:
        raise HTTPException(status_code=401, detail="กรุณาเข้าสู่ระบบก่อน")

    from grade_module import simulate_scenarios, format_scenario_text
    from agent import run_llm_direct

    result = simulate_scenarios(
        current_gpax=body.current_gpax,
        credits_earned=body.credits_earned,
        remaining_sems=body.remaining_sems,
        credits_per_sem=body.credits_per_sem,
        sem_gpas=body.sem_gpas,
        total_credits_remaining=body.credits_remaining,
    )

    summary_text = format_scenario_text(result)
    ai_comment = run_llm_direct(
        f"สรุป scenario นี้ให้กระชับและให้คำแนะนำ 2-3 ประโยค:\n\n{summary_text}"
    )

    return {
        "steps":      result["steps"],
        "summary":    summary_text,
        "ai_comment": ai_comment,
    }