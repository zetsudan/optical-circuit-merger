from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pandas as pd
import io, re, json
from decimal import Decimal, getcontext
from typing import List, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------- FastAPI ----------
app = FastAPI(title="DWDM TSV to XLSX Merger")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Precise arithmetic ----------
getcontext().prec = 18
STEP = Decimal("0.0125")        # 12.5 GHz
GRID_LOW = Decimal("191.325")
GRID_HIGH = Decimal("196.125")

# ---------- Readers ----------
def read_custom_tsv_bytes(file_bytes: bytes) -> pd.DataFrame:
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            text = file_bytes.decode(enc, errors="replace"); break
        except Exception: continue
    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    lines = []; header_done = False
    for line in text.splitlines():
        if line.startswith("#"):
            if "\t" in line and not header_done:
                lines.append(line.lstrip("#").strip()); header_done = True
            continue
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    if not cleaned: return pd.DataFrame()
    first = cleaned.splitlines()[0]
    sep = "\t" if "\t" in first else (";" if ";" in first else ",")
    df = pd.read_csv(io.StringIO(cleaned), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- Column extractors / filters ----------
def extract_required(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if "#Type" in df.columns:
        df = df[df["#Type"].astype(str).str.upper().str.contains("LOCAL_OPTICAL_SNC", na=False)].copy()
    if "Circuit ID" not in df.columns:
        raise KeyError("Expected 'Circuit ID' column not found.")
    pb_col = next((c for c in df.columns if c.lower().strip() == "passband list"), None)
    if pb_col is None:
        raise KeyError("Expected 'passband list' column not found.")
    fs_col = next((c for c in df.columns if c.lower().strip() == "freq slot plan type"), None)
    label_col = next((c for c in df.columns if c.lower().strip() == "label"), None)

    keep = ["Circuit ID", pb_col] + ([fs_col] if fs_col else []) + ([label_col] if label_col else [])
    out = df[keep].copy()
    out = out.rename(columns={pb_col: "passband list"})
    if fs_col: out = out.rename(columns={fs_col: "Freq Slot Plan Type"})
    else: out["Freq Slot Plan Type"] = ""
    if label_col: out = out.rename(columns={label_col: "Label"})
    else: out["Label"] = ""
    return out

# ---------- Passband parsing ----------
def D(x) -> Decimal: return Decimal(str(x))

def parse_passband_list(pb) -> List[Tuple[Decimal, Decimal]]:
    s = str(pb).strip()
    if not s: return []
    segs: List[Tuple[Decimal, Decimal]] = []
    for chunk in s.split(":"):
        ch = chunk.strip()
        if not ch: continue
        if ch.startswith("(") and ch.endswith(")"): ch = ch[1:-1]
        parts = [p.strip() for p in ch.replace(";", ",").split(",") if p.strip()]
        if len(parts) < 2: continue
        lo = D(parts[0].replace(",", ".")); hi = D(parts[1].replace(",", "."))
        if hi < lo: lo, hi = hi, lo
        segs.append((lo, hi))
    return segs

# ---------- OCG rules ----------
OCG_TAIL = re.compile(r"-OCG(\d{1,2})$", re.IGNORECASE)

def classify_fs(fs_raw: str | None) -> str | None:
    s = str(fs_raw or "").upper().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "OCG CP 1" in s or re.search(r"\bFS OCG ?1\b", s): return "OCG"
    if "FS NONE" in s or s == "NONE": return "NONE"
    return None

def fmt_dec(x: Decimal) -> str:
    s = format(x, "f")
    if "." in s: s = s.rstrip("0").rstrip(".")
    return s

def derive_extra_segments(cid: str, fs_raw: str | None, segs: List[Tuple[Decimal, Decimal]]):
    m = OCG_TAIL.search(str(cid).strip())
    if not m: return []
    idx = int(m.group(1))
    if not (1 <= idx <= 16): return []
    mode = classify_fs(fs_raw)
    if mode != "OCG": return []
    extras = []
    for lo, hi in segs:
        lo2 = lo - 2 * STEP; hi2 = lo
        extras.append({"lo": lo2, "hi": hi2, "label": f"OCG{idx} + FS OCG ({fmt_dec(lo2)} - {fmt_dec(lo)})"})
    return extras

# ---------- Grid ----------
def build_grid() -> pd.DataFrame:
    rows = []; f = GRID_LOW
    while f + STEP <= GRID_HIGH:
        lo = f; hi = f + STEP; ce = (lo + hi) / Decimal("2")
        rows.append((float(lo), float(ce), float(hi), lo, hi))
        f = hi
    return pd.DataFrame(rows, columns=[
        "Lower Edge Freq (THz)", "Center Freq (THz)", "Higher Edge Freq (THz)", "_lo", "_hi"
    ])

def base_label_for_segments(cid: str, segs: List[Tuple[Decimal, Decimal]]) -> str:
    if not segs: return cid
    if len(segs) == 1:
        lo, hi = segs[0]; slices = int(((hi - lo) / STEP).to_integral_value())
        return f"{cid} ({fmt_dec(lo)}–{fmt_dec(hi)}) [{slices}]"
    lows = [lo for lo, _ in segs]; highs = [hi for _, hi in segs]
    total = sum(int(((hi - lo) / STEP).to_integral_value()) for lo, hi in segs)
    return f"{cid} ({len(segs)} bands; {fmt_dec(min(lows))}–{fmt_dec(max(highs))}) [{total}]"

# ---------- Column plan ----------
def _normalize_plan(files: List[UploadFile], plan_json: str | None):
    n = len(files)
    default_order = list(range(n))
    default_names = [(f.filename or f"Table {i+1}") for i, f in enumerate(files)]
    if not plan_json:
        return [{"file": files[i], "name": default_names[i]} for i in default_order]
    try: data = json.loads(plan_json)
    except Exception: data = {}
    order = data.get("order", default_order)
    order = [i for i in order if isinstance(i, int) and 0 <= i < n]
    seen = set(); order = [i for i in order if (i not in seen and not seen.add(i))] or default_order
    names = data.get("names", default_names)
    norm_names = []
    for pos, fi in enumerate(order):
        try: name = names[fi] if isinstance(names, list) and len(names) > fi else default_names[fi]
        except Exception: name = default_names[fi]
        name = str(name).strip() or default_names[fi]
        norm_names.append(name)
    return [{"file": files[fi], "name": norm_names[pos]} for pos, fi in enumerate(order)]

# ---------- Helper: annotate FREE runs with lengths ----------
def annotate_free_runs(texts: list[str], colors: list[str]) -> list[str]:
    out = texts[:]
    n = len(out); i = 0
    while i < n:
        if colors[i] == "FREE" and out[i] == "FREE":
            j = i
            while j < n and colors[j] == "FREE" and out[j] == "FREE":
                j += 1
            run_len = j - i
            for k in range(i, j):
                out[k] = f"FREE [{run_len}]"
            i = j
        else:
            i += 1
    return out

# ---------- Core assembly ----------
def assemble_result(colspecs: List[dict], include_label: bool = False) -> tuple[pd.DataFrame, dict]:
    grid = build_grid()
    result = grid[["Lower Edge Freq (THz)", "Center Freq (THz)", "Higher Edge Freq (THz)"]].copy()
    color_cols = {}

    for spec in colspecs:
        f = spec["file"]; colname = spec["name"]
        raw = f.file.read()
        df = read_custom_tsv_bytes(raw)
        req = extract_required(df)
        req["segments"] = req["passband list"].map(parse_passband_list)
        req["base_label"] = req.apply(lambda r: base_label_for_segments(r["Circuit ID"], r["segments"]), axis=1)

        sep_nl = chr(10)
        if include_label:
            req["display_label"] = req.apply(
                lambda r: (r["base_label"] + (sep_nl + str(r["Label"]).strip() if str(r["Label"]).strip() else "")),
                axis=1
            )
        else:
            req["display_label"] = req["base_label"]

        extras = []
        for _, r in req.iterrows():
            extras.extend(derive_extra_segments(r["Circuit ID"], r.get("Freq Slot Plan Type", ""), r["segments"]))

        texts = []; colors = []
        for _, slot in grid.iterrows():
            l, u = slot["_lo"], slot["_hi"]
            labs, cols = [], []

            for _, r in req.iterrows():
                for lo, hi in r["segments"]:
                    if (hi > l) and (lo < u):
                        labs.append(r["display_label"]); cols.append("BLUE"); break

            for ex in extras:
                lo, hi = ex["lo"], ex["hi"]
                if (hi > l) and (lo < u):
                    labs.append(ex["label"]); cols.append("YELLOW")

            if labs:
                uniq, seen, ucols = [], set(), []
                for lab, ctag in zip(labs, cols):
                    if lab not in seen:
                        uniq.append(lab); ucols.append(ctag); seen.add(lab)
                texts.append(", ".join(uniq))
                colors.append("YELLOW" if "YELLOW" in ucols else "BLUE")
            else:
                texts.append("FREE"); colors.append("FREE")

        # NEW: помечаем длины свободных блоков
        texts = annotate_free_runs(texts, colors)

        result[colname] = texts
        color_cols[colname] = colors

    return result, color_cols

# ---------- Excel writer with merges & colors ----------
def to_excel_merged_colored(df: pd.DataFrame, color_cols: dict) -> bytes:
    import xlsxwriter
    bio = io.BytesIO()
    wb = xlsxwriter.Workbook(bio, {"in_memory": True})
    ws = wb.add_worksheet("result")

    hdr_fmt = wb.add_format({"bold": True, "bg_color": "#FFC000", "align": "center", "valign": "vcenter", "border": 1})
    grid_hdr_fmt = wb.add_format({"bold": True, "bg_color": "#FFE699", "align": "center", "valign": "vcenter", "border": 1})
    fmt_blue   = wb.add_format({"bg_color": "#CCECFF", "valign": "vcenter", "border": 1, "text_wrap": True})
    fmt_green  = wb.add_format({"bg_color": "#C6EFCE", "valign": "vcenter", "border": 1, "text_wrap": True})
    fmt_yellow = wb.add_format({"bg_color": "#FFF2CC", "valign": "vcenter", "border": 1, "text_wrap": True})
    grid_fmt   = wb.add_format({"align": "center", "valign": "vcenter", "border": 1, "num_format": "0.######"})

    for c, name in enumerate(df.columns):
        ws.write(0, c, name, grid_hdr_fmt if c < 3 else hdr_fmt)
    ws.set_column(0, 2, 16)
    ws.set_column(3, df.shape[1]-1, 86)

    nrows = df.shape[0]
    for r in range(1, nrows+1):
        for c in range(0, 3):
            ws.write_number(r, c, float(df.iat[r-1, c]), grid_fmt)

    def apply_merges(col_idx: int, col_name: str):
        texts = df[col_name].tolist()
        colors = color_cols[col_name]
        r = 1
        while r <= nrows:
            val = texts[r-1]; tag = colors[r-1]
            run_start = r; run_end = r
            while run_end + 1 <= nrows and texts[run_end] == val and colors[run_end] == tag:
                run_end += 1
            fmt = fmt_green if tag == "FREE" else (fmt_yellow if tag == "YELLOW" else fmt_blue)
            if run_end > run_start:
                ws.merge_range(run_start, col_idx, run_end, col_idx, val, fmt)
            else:
                ws.write(r, col_idx, val, fmt)
            r = run_end + 1

    for c in range(3, df.shape[1]):
        apply_merges(c, df.columns[c])

    wb.close()
    bio.seek(0)
    return bio.getvalue()

# ---------- Routes ----------
@app.get("/")
def index():
    p = Path(__file__).parent / "static" / "index.html"
    if p.exists(): return FileResponse(p)
    return HTMLResponse("<h1>Static index not found</h1>", status_code=500)

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.post("/preview", response_class=HTMLResponse)
async def preview(
    files: List[UploadFile] = File(...),
    start: float | None = None,
    rows: int = 150,
    plan: str | None = Form(None),
    include_label: int | None = Form(None),
):
    colspecs = _normalize_plan(files, plan)
    df, color_cols = assemble_result(colspecs, include_label=bool(include_label))
    rows = max(1, min(1500, int(rows)))
    if start is not None:
        try:
            base = float(GRID_LOW); step = float(STEP)
            start_idx = int(round((float(start) - base) / step))
            start_idx = max(0, min(len(df), start_idx))
        except Exception:
            start_idx = 0
        window = df.iloc[start_idx:start_idx+rows]
    else:
        window = df.head(rows)
    return window.to_html(index=False, escape=False)

@app.post("/merge")
async def merge(
    files: List[UploadFile] = File(...),
    plan: str | None = Form(None),
    include_label: int | None = Form(None),
):
    colspecs = _normalize_plan(files, plan)
    df, color_cols = assemble_result(colspecs, include_label=bool(include_label))
    xls = to_excel_merged_colored(df, color_cols)
    now = datetime.now(ZoneInfo("Asia/Yerevan"))
    fname = f"optical_circuits_merge_{now:%H%M}_{now:%d%m%Y}.xlsx"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(io.BytesIO(xls),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers)
