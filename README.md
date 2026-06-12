# UniAssist AI
### Intelligent Student Support and Advisor Monitoring System
> ระบบผู้ช่วยอัจฉริยะเพื่อการเรียนรู้และการติดตามนักศึกษาในระดับมหาวิทยาลัย

---

## ภาพรวมโปรเจค

UniAssist AI คือระบบ AI สำหรับนักศึกษาและอาจารย์ที่ปรึกษา ครอบคลุม 4 สาขาวิชา ได้แก่ IT, DSBA, BIT และ AIT โดยผสานเทคโนโลยี RAG + LLM Reasoning เข้ากับระบบคำนวณเกรดและแนะนำรายวิชา เพื่อลดภาระการค้นหาข้อมูลและช่วยให้นักศึกษาวางแผนการเรียนได้อย่างมีประสิทธิภาพ


## วัตถุประสงค์

- **ศึกษา** แหล่งข้อมูลวิชาการในมหาวิทยาลัยและแนวคิด AI Education Agent
- **พัฒนา AI Chatbot** ด้วย RAG + LLM Reasoning สำหรับตอบคำถามวิชาการ
- **พัฒนาระบบคำนวณ GPA** พร้อมประเมินความเสี่ยงทางการศึกษา
- **พัฒนาระบบแนะนำรายวิชา** แบบ Prerequisite-aware
- **พัฒนาระบบติดตามนักศึกษา** สำหรับอาจารย์ที่ปรึกษา

---

## Features

### 1. AI Chatbot (RAG-Based)
ตอบคำถามวิชาการจากเอกสารหลักสูตรและระเบียบการศึกษา ครอบคลุม 7 หมวดหมู่:

| หมวดหมู่ | ROUGE-L | LLM Judge |
|---|---|---|
| Honors (เกียรตินิยม) | 0.44 | 4.7 / 5 ⭐ |
| GPA | 0.14 | 4.5 / 5 |
| Assessment (การวัดผล) | 0.00 | 4.0 / 5 |
| Graduation (สำเร็จการศึกษา) | 0.26 | 4.0 / 5 |
| Prerequisite | 0.72 | 4.0 / 5 |
| Dismissal (พ้นสภาพ) | 0.32 | 3.0 / 5 |
| Registration (ลงทะเบียน) | 0.37 | 2.7 / 5 |
| **ภาพรวม** | **0.34** | **3.97 / 5** |

### 2. Grade Analysis Module
คำนวณ GPA และประเมินสถานะนักศึกษาจาก Transcript

| ตัวชี้วัด | ผลลัพธ์ |
|---|---|
| GPA Accuracy | **100%** |
| Risk Accuracy | **100%** |
| MAE | 0.0000 |
| F1-Score | 1.000 |
| Latency | 0.005 ms |

สถานะที่ระบบประเมิน: **ปกติ / Probation / พ้นสภาพ**

### 3. Course Recommendation System
แนะนำรายวิชาพร้อมตรวจสอบ Prerequisite และตารางเรียนอัตโนมัติ

| ตัวชี้วัด | ผลลัพธ์ |
|---|---|
| Schedule Conflict | **0%** (หลีกเลี่ยงตารางชนได้ 100%) |
| Prerequisite Compliance | 75% |
| Precision@5 | 0.56 |
| LLM Judge Score | 2.75 / 5 |
| Latency | 34.39 วินาที |

### 4. Web Application (4 หน้าหลัก)
- **Login** — ระบบ JWT + bcrypt
- **AI Chatbot** — สนทนาและอัปโหลด Transcript
- **Grade Analysis** — วิเคราะห์ผลการเรียน
- **GPA Simulator** — จำลองสถานการณ์การเรียน

---

## System Architecture

```
INPUT
├── เว็บทะเบียน + PDF หลักสูตร (IT / DSBA / BIT / AIT)
├── คำถามข้อความ (Text Query) ผ่าน Chatbot
└── Transcript + ตารางเรียน
         │
         ▼
AI SYSTEM
├── RAG / MRAG + LLM Reasoning
├── Vector DB (ChromaDB) — Semantic Search
│     └── Embedding: paraphrase-multilingual-MiniLM-L12-v2
├── Prerequisite Graph (SQLite)
└── ReAct Agent: Thought → Action → Observation
      └── 5 Tools:
          ├── search_curriculum_rules
          ├── query_teach_table
          ├── simulate_gpa_scenario
          ├── query_prerequisite_graph
          └── recommend_courses
         │
         ▼
OUTPUT
├── Chatbot ตอบคำถามวิชาการ
├── GPA + สถานะความเสี่ยง + คำแนะนำ
└── รายวิชาแนะนำ (ตรวจ Conflict อัตโนมัติ)
```

---

## Tech Stack

| ส่วน | เทคโนโลยี |
|---|---|
| Backend | Python + FastAPI |
| Frontend | Web Application |
| Vector Database | ChromaDB |
| Relational Database | SQLite |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 |
| LLM | Typhoon v2.5 (30B-A3B) |
| Security | JWT + bcrypt |
| Evaluation | Groq AI (LLM as Judge) |
| Architecture | Client–Server |

---

## วิธีการดำเนินการ

```
01. รวบรวมข้อมูล
    เว็บทะเบียน + PDF หลักสูตร 4 สาขา
    แปลงด้วย Typhoon-v2.5-30b-a3b-instruct
    แบ่งเป็น Unstructured (ข้อความ) + Structured (ตาราง)

02. นำเข้าฐานข้อมูล
    ข้อความ → Embedding → ChromaDB (Semantic Search)
    ตาราง → SQLite (Prerequisite Graph)

03. พัฒนา AI Agent (ReAct)
    วนซ้ำ Thought → Action → Observation
    ไม่สร้างคำตอบจากความจำ — เรียก Tool ทุกครั้ง

04. Grade Module
    Transcript → Rule Engine → GPA Calculator
    → Status Evaluator → Scenario Simulation → LLM Summary

05. Web Application
    Chatbot + Grade Analysis + GPA Simulator

06. ประเมินผล
    ROUGE-L / LLM as Judge / Accuracy / F1 / Latency
```

---

## ข้อจำกัดและความท้าทาย

**AI Chatbot**
- ROUGE-L = 0.0 สำหรับคำถามประเภท Registration
- Latency เฉลี่ย 15.3 วินาที / P95 สูงถึง 25.7 วินาที
- ReAct loop ต้องประสาน Vector DB + SQLite พร้อมกัน — โอกาสเลือก Tool ผิดสูง

**Course Recommendation**
- Prerequisite Compliance ทำได้แค่ 75% (เป้าหมาย 100%)
- Latency สูงถึง 34.4 วินาที

**ข้อมูล**
- PDF หลักสูตรมีจำนวนมาก (IT = 443 หน้า) และ format ไม่สม่ำเสมอ
- ต้นทุน LLM API สูงในขั้น ReAct loop

---

## แผนการพัฒนาในอนาคต

- [ ] Advisor Dashboard สำหรับอาจารย์ที่ปรึกษา
- [ ] ระบบ Scholarship แนะนำทุนการศึกษา
- [ ] User Study & Feedback จากนักศึกษาจริง
- [ ] ปรับปรุง Prerequisite Graph ให้ Compliance ถึง 100%
- [ ] ลด Latency ของ Course Recommender

---

## Related Research

1. Systematic Analysis of RAG-Based LLMs for Medical Chatbot
2. RAG-Based AI Chatbot for Student and Institutional Assistance
3. Development of an Academic Services Chatbot Based on RAG
4. LLM-Powered Teaching Assistants (Intent-based → LLM TA)
5. Typhoon v2.5 — Thai LLM
6. OpenThaiGPT 1.5
7. Qwen3 Technical Report

---

## ขอบเขตข้อมูล

ระบบครอบคลุม 4 สาขาวิชาของสถาบัน:

| สาขา | ย่อ |
|---|---|
| Information Technology | IT |
| Data Science and Business Analytics | DSBA |
| Business Information Technology | BIT |
| Artificial Intelligence Technology | AIT |
