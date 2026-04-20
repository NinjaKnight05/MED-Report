import streamlit as st
import os, re, shutil, tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MedReport AI", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.stApp { background:#0d1117; color:#e6edf3; }
.block-container { padding-top:1.5rem !important; }
.med-header {
    background:linear-gradient(135deg,#0d2137 0%,#0a3a5c 60%,#0e4f7a 100%);
    border:1px solid #1f6aa5; border-radius:14px;
    padding:1.5rem 2rem; margin-bottom:1.5rem;
}
.med-header h1 { margin:0; font-size:1.9rem; color:#e6edf3; }
.med-header p  { margin:.4rem 0 0; color:#8db4d4; font-size:.95rem; }
.pat-card {
    background:#161b22; border:1px solid #21262d; border-radius:12px;
    padding:1rem 1.25rem; margin-bottom:1.25rem;
    display:flex; flex-wrap:wrap; gap:14px;
}
.pat-item { font-size:.82rem; }
.pat-item .lbl { color:#8b949e; }
.pat-item .val { color:#e6edf3; font-weight:600; }
.alert-box {
    background:#1f1215; border:1px solid #5a1d1d;
    border-left:4px solid #f85149; border-radius:10px;
    padding:1rem 1.25rem; margin-bottom:1.25rem;
}
.alert-title { color:#f85149; font-weight:700; font-size:.85rem;
               text-transform:uppercase; letter-spacing:.06em; margin-bottom:.6rem; }
.alert-item {
    display:grid; grid-template-columns:1fr auto; gap:8px; align-items:start;
    padding:4px 0; border-bottom:1px solid #2d1a1a; font-size:.83rem; color:#ffa198;
}
.alert-item:last-child { border-bottom:none; }
.alert-item .why { color:#8b949e; font-size:.78rem; }
.sec-wrap { margin-bottom:1.25rem; }
.sec-head {
    font-size:.75rem; font-weight:600; color:#58a6ff;
    text-transform:uppercase; letter-spacing:.07em;
    padding-bottom:6px; border-bottom:1px solid #21262d; margin-bottom:4px;
}
.tbl-hdr {
    display:grid; grid-template-columns:2.2fr 1fr 1.4fr .5fr;
    gap:6px; padding:4px 6px;
    font-size:.72rem; font-weight:600; color:#8b949e; text-transform:uppercase;
}
.tbl-row {
    display:grid; grid-template-columns:2.2fr 1fr 1.4fr .5fr;
    gap:6px; padding:6px 6px; border-bottom:1px solid #161b22;
    align-items:center; font-size:.83rem;
}
.tbl-row:last-child { border-bottom:none; }
.tbl-row:hover { background:#161b22; border-radius:6px; }
.tname { color:#c9d1d9; }
.tval  { color:#e6edf3; font-weight:600; }
.tref  { color:#8b949e; font-size:.78rem; }
.st-ok   { color:#3fb950; font-size:.8rem; font-weight:700; }
.st-high { color:#f85149; font-size:.8rem; font-weight:700; }
.st-low  { color:#58a6ff; font-size:.8rem; font-weight:700; }
.st-brd  { color:#d29922; font-size:.8rem; font-weight:700; }
.st-abn  { color:#f85149; font-size:.8rem; font-weight:700; }
.bubble-user {
    background:#0d2137; border:1px solid #1f6aa5;
    border-radius:14px 14px 4px 14px;
    padding:.7rem 1rem; margin:.4rem 0; color:#cce5ff; font-size:.88rem;
}
.bubble-ai {
    background:#161b22; border:1px solid #21262d; border-left:3px solid #58a6ff;
    border-radius:14px 14px 14px 4px;
    padding:.7rem 1rem; margin:.4rem 0; color:#e6edf3; font-size:.88rem; line-height:1.6;
}
.src-chip {
    display:inline-block; background:#0d2137; color:#58a6ff;
    font-size:.72rem; padding:2px 9px; border-radius:20px;
    margin:3px 2px; border:1px solid #1f6aa5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="med-header">
    <h1>🩺 MedReport AI</h1>
    <p>Upload patient PDF lab reports → instant structured summary + Q&amp;A</p>
</div>""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k in ["rag_chain","retriever","chat_history","summary_html","raw_text","index_dir"]:
    if k not in st.session_state:
        st.session_state[k] = [] if k == "chat_history" else None

# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_text(text):
    if not text or not isinstance(text, str):
        return None
    text = text.encode("utf-8","ignore").decode("utf-8","ignore")
    text = re.sub(r'\s+',' ',text).strip()
    text = re.sub(r'page \d+','',text,flags=re.IGNORECASE)
    return text if text.strip() else None


def extract_full_text(uploaded_files):
    tmp_dir = tempfile.mkdtemp()
    raw_pages, all_docs = [], []
    for uf in uploaded_files:
        pdf_path = os.path.join(tmp_dir, uf.name)
        with open(pdf_path,"wb") as f:
            f.write(uf.read())
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()
        raw_pages.extend([p.page_content for p in pages])
        all_docs.extend(pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    cleaned = []
    for doc in chunks:
        c = clean_text(doc.page_content)
        if c:
            doc.page_content = c
            cleaned.append(doc)
    return tmp_dir, raw_pages, cleaned


def build_rag(cleaned, tmp_dir):
    index_dir = os.path.join(tmp_dir, "faiss_index")
    emb = OllamaEmbeddings(model="mxbai-embed-large:latest")
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    db = FAISS.from_documents(cleaned, emb)
    db.save_local(index_dir)
    retriever = db.as_retriever(
        search_kwargs={"k":4,"fetch_k":12,"lambda_mult":0.85},
        search_type="mmr",
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are an expert medical assistant analyzing lab reports.
Use ONLY the provided context. Do NOT use outside knowledge.
If information is not found say: "Not found in available reports."
Flag abnormal values with ⚠️. Never provide a diagnosis.
Context:\n{context}"""),
        ("human","{question}"),
    ])
    llm = ChatOllama(model="phi3:mini", temperature=0.1)
    def fmt(docs): return "\n\n".join(d.page_content for d in docs)
    chain = (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever, index_dir


# ── Category definitions ───────────────────────────────────────────────────────
CATEGORIES = [
    ("Patient Info",
     "Extract: Name, Age, Sex, Date of Birth, Lab ID, Collection Date, "
     "ABO Type, Rh Type. Return as 'Key: Value' lines only. Nothing else."),

    ("Complete Blood Count",
     "List ONLY these CBC tests if present: Hemoglobin, RBC Count, Hematocrit, "
     "MCV, MCH, MCHC, RDW CV, WBC Count, Neutrophils, Lymphocytes, Eosinophils, "
     "Monocytes, Basophils, Platelet Count, MPV, ESR, RBC Morphology, Malarial Parasite. "
     "Each line MUST be exactly: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS\n"
     "STATUS must be one word: NORMAL, HIGH, LOW, BORDERLINE, ABNORMAL, NEGATIVE, or POSITIVE."),

    ("Diabetes Markers",
     "List ONLY: Fasting Blood Sugar, HbA1c, Mean Blood Glucose, Urine Glucose, Microalbumin. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Lipid Profile",
     "List ONLY: Cholesterol, Triglyceride, HDL Cholesterol, Direct LDL, VLDL, "
     "CHOL/HDL Ratio, LDL/HDL Ratio. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Thyroid Function",
     "List ONLY: T3 Triiodothyronine, T4 Thyroxine, TSH. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Vitamins & Iron Studies",
     "List ONLY: Vitamin D 25(OH), Vitamin B12, Iron, TIBC, Transferrin Saturation. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Biochemistry",
     "List ONLY: Creatinine Serum, Urea, Blood Urea Nitrogen, Uric Acid, Calcium, "
     "SGPT, SGOT, Total Protein, Albumin, Globulin, Total Bilirubin, Conjugated Bilirubin, "
     "Sodium, Potassium, Chloride. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Immunology & Cardiovascular",
     "List ONLY: Homocysteine Serum, IgE, PSA Prostate Specific Antigen. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Infectious Disease Screening",
     "List ONLY: HIV I II Ab Ag P24, HBsAg. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS\n"
     "STATUS = NEGATIVE if non-reactive, POSITIVE if reactive."),

    ("Haemoglobin Electrophoresis",
     "List ONLY: Hb A, Hb A2, Foetal Hb, P2 Peak, P3 Peak, Interpretation. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),

    ("Urinalysis",
     "List ONLY: Colour, Clarity, pH, Specific Gravity, Urine Glucose, Urine Protein, "
     "Bilirubin, Urobilinogen, Urine Ketone, Nitrite, Pus Cells, Red Cells, "
     "Epithelial Cells, Casts, Crystals. "
     "Format: TEST NAME | RESULT UNIT | REFERENCE RANGE | STATUS"),
]


def ask_llm_category(llm, context_text, instruction):
    msgs = [
        {"role":"system","content":
         "You are a medical data extractor. Return ONLY the formatted lines requested. "
         "No preamble, no markdown, no explanation. "
         "Use pipe | separator. If a test is missing write: NOT FOUND"},
        {"role":"user","content":
         f"LAB REPORT TEXT:\n{context_text[:7000]}\n\nINSTRUCTION:\n{instruction}"}
    ]
    return llm.invoke(msgs).content.strip()


def parse_pipe_lines(raw: str):
    rows = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.upper() == "NOT FOUND":
            continue
        # skip header lines
        if line.lower().startswith("test name"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0]:
            rows.append({
                "name":   parts[0],
                "val":    parts[1] if len(parts) > 1 else "—",
                "ref":    parts[2] if len(parts) > 2 else "—",
                "status": parts[3] if len(parts) > 3 else "NORMAL",
            })
    return rows


def parse_patient_info(raw: str) -> dict:
    info = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            info[k.strip().lower()] = v.strip()
    return info


def status_class(s: str) -> str:
    s = s.strip().upper()
    if s in ("HIGH","H","ELEVATED","ABOVE","ABNORMAL HIGH","PRESENT","POSITIVE"): return "st-high"
    if s in ("LOW","L","BELOW","DEFIC","DEFICIENT"):                               return "st-low"
    if s in ("BORDERLINE","BORDER","BRD","NEAR"):                                  return "st-brd"
    if s in ("ABNORMAL","ABN","ABNORMAL"):                                         return "st-abn"
    return "st-ok"  # NORMAL, NEGATIVE, OK, WITHIN, NR etc


def status_label(s: str) -> str:
    cls = status_class(s)
    return {"st-high":"▲ HIGH","st-low":"▼ LOW","st-brd":"~ BORDER",
            "st-ok":"✓","st-abn":"⚠ ABN"}.get(cls, s[:8])


def rows_to_html(rows):
    if not rows:
        return '<div style="font-size:.8rem;color:#8b949e;padding:4px 6px;">No data found.</div>'
    html = ('<div class="tbl-hdr">'
            '<span>Test</span><span>Result</span>'
            '<span>Reference</span><span>Status</span></div>')
    for r in rows:
        sc  = status_class(r["status"])
        lbl = status_label(r["status"])
        html += (
            f'<div class="tbl-row">'
            f'<div class="tname">{r["name"]}</div>'
            f'<div class="tval">{r["val"]}</div>'
            f'<div class="tref">{r["ref"]}</div>'
            f'<div class="{sc}">{lbl}</div>'
            f'</div>'
        )
    return html


def patient_card_html(info: dict) -> str:
    abo = info.get("abo type","")
    rh  = info.get("rh type","")
    bg  = f"{abo} {rh}".strip() or "—"
    fields = [
        ("Name",        info.get("name","—")),
        ("Sex / Age",   f"{info.get('sex','—')} / {info.get('age','—')}"),
        ("DOB",         info.get("date of birth") or info.get("dob","—")),
        ("Lab ID",      info.get("lab id","—")),
        ("Collected",   info.get("collection date") or info.get("collected on","—")),
        ("Blood Group", bg),
    ]
    items = "".join(
        f'<div class="pat-item"><span class="lbl">{k}&nbsp;</span>'
        f'<span class="val">{v}</span></div>'
        for k,v in fields
    )
    return f'<div class="pat-card">{items}</div>'


def alert_box_html(cat_data: dict) -> str:
    skip = {"Patient Info"}
    flagged = []
    for cat, rows in cat_data.items():
        if cat in skip: continue
        for r in rows:
            s = r["status"].strip().upper()
            if s not in ("NORMAL","OK","WITHIN","WNL","NEGATIVE","NON REACTIVE",
                         "NEG","NR","NOT FOUND",""):
                flagged.append(r)
    if not flagged:
        return ""
    items = "".join(
        f'<div class="alert-item">'
        f'<div><strong>{r["name"]}</strong>: {r["val"]}&nbsp;'
        f'<span class="why">(ref: {r["ref"]})</span></div>'
        f'<div class="{status_class(r["status"])}">{status_label(r["status"])}</div>'
        f'</div>'
        for r in flagged
    )
    return (f'<div class="alert-box">'
            f'<div class="alert-title">⚠ Abnormal values requiring attention</div>'
            f'{items}</div>')


def generate_summary_html(raw_pages):
    llm = ChatOllama(model="phi3:mini", temperature=0.0)
    combined = "\n".join(raw_pages)

    pbar = st.progress(0, text="Extracting patient details...")

    # Patient info
    pat_raw  = ask_llm_category(llm, combined, CATEGORIES[0][1])
    pat_info = parse_patient_info(pat_raw)
    pat_html = patient_card_html(pat_info)

    # Categories
    cat_data = {}
    sec_html = ""
    total = len(CATEGORIES) - 1
    for i, (cat_name, instruction) in enumerate(CATEGORIES[1:], 1):
        pbar.progress(int(i/total * 90), text=f"Extracting: {cat_name}...")
        raw  = ask_llm_category(llm, combined, instruction)
        rows = parse_pipe_lines(raw)
        cat_data[cat_name] = rows
        sec_html += (
            f'<div class="sec-wrap">'
            f'<div class="sec-head">{cat_name}</div>'
            f'{rows_to_html(rows)}</div>'
        )

    pbar.progress(98, text="Building summary...")
    alert_html = alert_box_html(cat_data)
    pbar.progress(100, text="Done!")
    pbar.empty()

    return pat_html + alert_html + sec_html


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Upload Reports")
    uploaded_files = st.file_uploader(
        "Drop PDF lab reports here",
        type=["pdf"], accept_multiple_files=True,
    )
    process_btn = st.button("⚙️ Process Reports", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("📄 Reading & indexing PDFs..."):
                tmp_dir, raw_pages, cleaned = extract_full_text(uploaded_files)
            if not cleaned:
                st.error("No usable text found in PDFs.")
            else:
                with st.spinner("🔢 Building vector index..."):
                    chain, retriever, index_dir = build_rag(cleaned, tmp_dir)
                st.session_state.rag_chain    = chain
                st.session_state.retriever    = retriever
                st.session_state.index_dir    = index_dir
                st.session_state.raw_text     = raw_pages
                st.session_state.chat_history = []
                st.session_state.summary_html = None
                st.success(f"✅ {len(uploaded_files)} file(s) ready!")

    st.markdown("---")
    if st.session_state.rag_chain:
        if st.button("🧹 Clear Session", use_container_width=True):
            for k in ["rag_chain","retriever","chat_history",
                      "summary_html","raw_text","index_dir"]:
                st.session_state[k] = [] if k == "chat_history" else None
            st.rerun()

# ── Main ────────────────────────────────────────────────────────────────────────
if not st.session_state.rag_chain:
    st.info("👈 Upload PDF lab reports in the sidebar and click **Process Reports**.")
    st.stop()

# Generate summary once
if st.session_state.summary_html is None:
    st.session_state.summary_html = generate_summary_html(st.session_state.raw_text)

st.markdown("## 📋 Lab Report Summary")
st.markdown(st.session_state.summary_html, unsafe_allow_html=True)
st.markdown("---")

# ── Chat ────────────────────────────────────────────────────────────────────────
st.markdown("## 💬 Ask Questions About the Reports")

for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="bubble-user">🧑&nbsp; {msg["content"]}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="bubble-ai">🤖&nbsp; {msg["content"]}</div>',
            unsafe_allow_html=True)
        if msg.get("sources"):
            chips = "".join(
                f'<span class="src-chip">📄 {os.path.basename(s)}</span>'
                for s in msg["sources"]
            )
            st.markdown(chips, unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    c1, c2 = st.columns([5, 1])
    with c1:
        user_q = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="e.g. Is my HbA1c under control? What does high homocysteine mean?"
        )
    with c2:
        send = st.form_submit_button("Send →", use_container_width=True)

if send and user_q.strip():
    st.session_state.chat_history.append({"role":"user","content":user_q})
    with st.spinner("Thinking..."):
        answer    = st.session_state.rag_chain.invoke(user_q)
        retrieved = st.session_state.retriever.invoke(user_q)
        sources   = list({d.metadata.get("source","unknown") for d in retrieved})
    st.session_state.chat_history.append(
        {"role":"assistant","content":answer,"sources":sources})
    st.rerun()