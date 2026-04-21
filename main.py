"""
AI-Powered Financial Investigation System
==========================================
FastAPI backend for analyzing bank statements / transaction files
(CSV / Excel / PDF) and producing an investigator-grade report.

Pipeline:
    File upload  ->  Pandas extraction  ->  AI summary (OpenAI)  ->  JSON report

Endpoints:
    GET  /              -> service info
    GET  /health        -> liveness probe
    POST /analyze       -> upload file, get full investigation report

Author: Cyber Forensic AI
"""

from __future__ import annotations

import io
import os
import re
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import pdfplumber  # type: ignore
except ImportError:  # pragma: no cover
    pdfplumber = None

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("forensic-api")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Financial Investigation System",
    description="Cyber Forensic backend for fraud / suspicious transaction analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Android app talks directly; tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TopItem(BaseModel):
    value: str
    count: int


class AnalysisResponse(BaseModel):
    file_name: str
    total_rows: int
    total_columns: int
    columns_detected: List[str]
    top_accounts: List[TopItem]
    top_banks: List[TopItem]
    most_active_accounts: List[TopItem]
    top_utrs: List[TopItem]
    transaction_date_range: Optional[Dict[str, str]] = None
    suspicious_indicators: List[str]
    ai_summary: str
    risk_level: str


# ---------------------------------------------------------------------------
# Helpers - file -> DataFrame
# ---------------------------------------------------------------------------
def _read_csv(data: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=400, detail="Unable to decode CSV file.")


def _read_excel(data: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), engine="openpyxl")


def _read_pdf(data: bytes) -> pd.DataFrame:
    if pdfplumber is None:
        raise HTTPException(status_code=500, detail="pdfplumber not installed.")
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables() or []:
                if not table:
                    continue
                if header is None:
                    header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(table[0])]
                    rows.extend(table[1:])
                else:
                    rows.extend(table)
    if not rows or header is None:
        raise HTTPException(status_code=400, detail="No tabular data found in PDF.")
    width = len(header)
    cleaned = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in rows]
    return pd.DataFrame(cleaned, columns=header)


def parse_file(filename: str, content: bytes) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        df = _read_csv(content)
    elif name.endswith((".xlsx", ".xls")):
        df = _read_excel(content)
    elif name.endswith(".pdf"):
        df = _read_pdf(content)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Allowed: .csv, .xlsx, .xls, .pdf",
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Helpers - column detection
# ---------------------------------------------------------------------------
ACCOUNT_PATTERNS = ["account", "acct", "a/c", "acc no", "account no", "account number"]
BANK_PATTERNS    = ["bank", "ifsc", "branch"]
UTR_PATTERNS     = ["utr", "ref no", "reference", "txn id", "transaction id", "rrn"]
DATE_PATTERNS    = ["date", "txn date", "transaction date", "value date", "posting date"]
AMOUNT_PATTERNS  = ["amount", "amt", "credit", "debit", "withdrawal", "deposit", "value"]


def _find_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    # exact match first
    for p in patterns:
        if p in lower:
            return lower[p]
    # contains match
    for col_l, col_orig in lower.items():
        if any(p in col_l for p in patterns):
            return col_orig
    return None


def _top_n(series: pd.Series, n: int = 10) -> List[TopItem]:
    series = series.dropna().astype(str).str.strip()
    series = series[series != ""]
    counts = series.value_counts().head(n)
    return [TopItem(value=str(idx), count=int(cnt)) for idx, cnt in counts.items()]


# ---------------------------------------------------------------------------
# Heuristic suspicious-pattern detector (runs even when AI is offline)
# ---------------------------------------------------------------------------
def detect_suspicious(df: pd.DataFrame) -> List[str]:
    flags: List[str] = []

    acc_col = _find_col(df, ACCOUNT_PATTERNS)
    amt_col = _find_col(df, AMOUNT_PATTERNS)
    date_col = _find_col(df, DATE_PATTERNS)

    if acc_col:
        counts = df[acc_col].dropna().astype(str).value_counts()
        hyper = counts[counts >= 20]
        if not hyper.empty:
            flags.append(
                f"{len(hyper)} account(s) appear 20+ times — possible layering / mule activity."
            )

    if amt_col:
        amounts = pd.to_numeric(
            df[amt_col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False),
            errors="coerce",
        ).dropna()
        if not amounts.empty:
            high = amounts[amounts >= 100000]
            if len(high) > 0:
                flags.append(f"{len(high)} high-value transaction(s) >= 1,00,000 detected.")
            round_txns = amounts[(amounts % 10000 == 0) & (amounts >= 10000)]
            if len(round_txns) >= 5:
                flags.append(
                    f"{len(round_txns)} suspiciously round transactions (multiples of 10,000)."
                )

    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna()
        if not dates.empty:
            same_day = dates.dt.date.value_counts()
            spikes = same_day[same_day >= 10]
            if not spikes.empty:
                flags.append(
                    f"{len(spikes)} day(s) with 10+ transactions — possible burst activity."
                )

    if not flags:
        flags.append("No strong heuristic red flags detected at the surface level.")
    return flags


def estimate_risk(flags: List[str], df: pd.DataFrame) -> str:
    score = 0
    for f in flags:
        if "mule" in f or "layering" in f:
            score += 3
        if "high-value" in f:
            score += 2
        if "round transactions" in f:
            score += 2
        if "burst" in f:
            score += 2
    if len(df) > 5000:
        score += 1
    if score >= 6:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# AI summary
# ---------------------------------------------------------------------------
def ai_summary(stats: Dict[str, Any]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return (
            "AI summary unavailable (OPENAI_API_KEY not configured). "
            "Heuristic analysis is shown above. Set OPENAI_API_KEY in environment "
            "to enable GPT-powered investigator narrative."
        )
    try:
        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        prompt = (
            "You are a senior cyber-forensic financial investigator assisting "
            "police / cyber-crime cell. Given the structured statistics below from "
            "a bank statement, write a concise investigator-style report (max 250 "
            "words). Cover: 1) overview, 2) suspicious accounts / banks, 3) likely "
            "fraud patterns (layering, mule, structuring), 4) recommended next "
            "investigation steps. Be factual, professional, no markdown headings.\n\n"
            f"STATISTICS:\n{stats}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=600,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:  # pragma: no cover
        log.exception("OpenAI call failed")
        return f"AI summary failed: {e}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {
        "service": "AI Financial Investigation System",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 25 MB).")

    log.info("Analyzing %s (%d bytes)", file.filename, len(content))
    df = parse_file(file.filename, content)

    acc_col  = _find_col(df, ACCOUNT_PATTERNS)
    bank_col = _find_col(df, BANK_PATTERNS)
    utr_col  = _find_col(df, UTR_PATTERNS)
    date_col = _find_col(df, DATE_PATTERNS)

    top_accounts = _top_n(df[acc_col]) if acc_col else []
    top_banks    = _top_n(df[bank_col]) if bank_col else []
    top_utrs     = _top_n(df[utr_col]) if utr_col else []
    most_active  = top_accounts[:5]

    date_range: Optional[Dict[str, str]] = None
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True).dropna()
        if not dates.empty:
            date_range = {
                "from": str(dates.min().date()),
                "to":   str(dates.max().date()),
            }

    flags = detect_suspicious(df)
    risk  = estimate_risk(flags, df)

    stats_for_ai: Dict[str, Any] = {
        "file_name": file.filename,
        "total_rows": int(len(df)),
        "columns": list(df.columns)[:30],
        "top_accounts": [t.model_dump() for t in top_accounts],
        "top_banks":    [t.model_dump() for t in top_banks],
        "top_utrs":     [t.model_dump() for t in top_utrs],
        "date_range": date_range,
        "suspicious_indicators": flags,
        "risk_level": risk,
    }
    summary = ai_summary(stats_for_ai)

    return AnalysisResponse(
        file_name=file.filename,
        total_rows=int(len(df)),
        total_columns=int(df.shape[1]),
        columns_detected=list(df.columns),
        top_accounts=top_accounts,
        top_banks=top_banks,
        most_active_accounts=most_active,
        top_utrs=top_utrs,
        transaction_date_range=date_range,
        suspicious_indicators=flags,
        ai_summary=summary,
        risk_level=risk,
    )


# ---------------------------------------------------------------------------
# Local dev entry-point (Render will use the start command from render.yaml)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
