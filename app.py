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
    """
    Reads TSV/CSV/TXT from bytes, stripping comment lines starting with '#'.
    Autodetects delimiter among: TAB / ';' / ','.
    """
    # decoding
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            text = file_bytes.decode(enc, errors="replace")
            break
        except Exception:
            continue
    if text is None:
        text = file_bytes.decode("utf-8", errors="replace")

    # strip comment lines, but keep commented header if present
    lines = []
    header_done = False
    for line in text.splitlines():
        if line.startswith("#"):
            if "\t" in line and not header_done:
                lines.append(line.lstrip("#").strip())
                header_done = True
            continue
        lines.append(line)

    cleaned = "\n".join(lines).strip()
    first = cleaned.splitlines()[0] if cleaned else ""
    if "\t" in first:
        sep = "\t"
    elif ";" in first:
        sep = ";"
    else:
        sep = ","
    df = pd.read_csv(io.StringIO(cleaned), sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- Column extractors / filters ----------
def extract_required(df: pd.DataFrame) -> pd.DataFrame:
    """
    - If '#Type' exists -> keep only rows with LOCAL_OPTICAL_SNC (case-insensitive).
    - Keep columns: Circuit ID, passband list, Freq Slot Plan Type (optional).
    """
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols

    # filter by #Type (LOCAL only)
    if "#Type" in df.columns:
        df = df[df["#Type"].astype(str).str.upper().str.contains("LOCAL_OPTICAL_SNC", na=False)].copy()

    if "Circuit ID" not in df.columns:
        raise KeyError("Expected 'Circuit ID' column not found.")

    # passband list column (case-insensitive exact name)
    pb_col = next((c for c in df.columns if c.lower().strip() == "passband list"), None)
    if pb_col is None:
        raise KeyError("Expected 'passband list' column not found.")

    # Freq Slot Plan Type (optional)
    fs_col = next((c for c in df.columns if c.lower().strip() == "freq slot plan type"), None)

    out = df[["Circuit ID", pb_col] + ([fs_col] if fs_col else [])].copy()
    out = out.rename(columns={pb_col: "passband list"})
    if fs_col:
        out = out.rename(columns={fs_col: "Freq Slot Plan Type"})
    else:
        out["Freq Slot Plan Type"] = ""
    return out

# ---------- Passband parsing ----------
def D(x) -> Decimal:
    return Decimal(str(x))

def parse_passband_list(pb) -> List[Tuple[Decimal, Decimal]]:
    """
    Supports multiple segments like:
      (193.9375,193.9625):(194.1375,194.1625):...
    Accepts semicolon/comma inside tuple, decimal comma tolerated.
    Returns list of Decimal tuples (low, high) with low <= high.
    """
    s = str(pb).strip()
    if not s:
        return []
    segs: List[Tuple[Decimal, Decimal]] = []
    for chunk in s.split(":"):
        ch = chunk.strip()
        if not ch:
            continue
        if ch.startswith("(") and ch.endswith(")"):
            ch = ch[1:-1]
        parts = [p.strip() for p in ch.replace(";", ",").split(",") if p.strip()]
        if len(parts) < 2:
            continue
        lo = D(parts[0].replace(",", "."))
        hi = D(parts[1].replace(",", "."))
        if hi < lo:
            lo, hi = hi, lo
        segs.append((lo, hi))
    return segs

# ---------- OCG rules ----------
OCG_TAIL = re.compile(r"-OCG(\d{1,2})$", re.IGNORECASE)

def classify_fs(fs_raw: str | None) -> str | None:
    """
    Normalize 'Freq Slot Plan Type':
      -> 'OCG'  for 'OCG-CP-1' or 'FS OCG-1' (any spacing/dashes/case)
      -> 'NONE' for 'FS NONE' or 'NONE'
      -> None   otherwise
    """
    s = str(fs_raw or "").upper().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if "OCG CP 1" in s or re.search(r"\bFS OCG ?1\b", s):
        return "OCG"
    if "FS NONE" in s or s == "NONE":
        return "NONE"
    return None

def fmt_dec(x: Decimal) -> str:
    s = format(x, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def derive_extra_segments(cid: str, fs_raw: str | None, segs: List[Tuple[Decimal, Decimal]]):
    """
    Add extra segments ONLY for OCG mode:
      - If Circuit ID ends with -OCG1..OCG16 and FS is 'OCG' → add 2 slots BELOW each base segment:
            [lo-0.025, lo)
        Label: 'OCGx + FS OCG (lo-0.025 - lo)'
      - If FS is 'NONE' → add nothing (new rule).
    Return list of dicts: {"lo": Decimal, "hi": Decimal, "label": str}
    """
    m = OCG_TAIL.search(str(cid).strip())
    if not m:
        return []
    idx = int(m.group(1))
    if not (1 <= idx <= 16):
        return []
    mode = classify_fs(fs_raw)
    if mode != "OCG":  # FS NONE or unknown → do not add
        return []

    extras = []
    for lo, hi in segs:
        lo2 = lo - 2 * STEP
        hi2 = lo                 # [lo-0.025, lo)
        extras.append({
            "lo": lo2,
            "hi": hi2,
            "label": f"OCG{idx} + FS OCG ({fmt_dec(lo2)} - {fmt_dec(lo)})",
        })
    return extras

# ---------- Grid & formatting ----------
def build_grid() -> pd.DataFrame:
    rows = []
    f = GRID_LOW
    while f + STEP <= GRID_HIGH:
        lo = f
        hi = f + STEP
        ce = (lo + hi) / Decimal("2")
        rows.append((float(lo), float(ce), float(hi), lo, hi))
        f = hi
    return pd.DataFrame(rows, columns=[
        "Lower Edge Freq (THz)", "Center Freq (THz)", "Higher Edge Freq (THz)",
        "_lo", "_hi"  # Decimal intervals for robust overlap
    ])

def base_label_for_segments(cid: str, segs: List[Tuple[Decimal, Decimal]]) -> str:
    if not segs:
        return cid
    if len(segs) == 1:
        lo, hi = segs[0]
        slices = int(((hi - lo) / STEP).to_integral_value())
        return f"{cid} ({fmt_dec(lo)}–{fmt_dec(hi)}) [{slices}]"
    lows = [lo for lo, _ in segs]
    highs = [hi for _, hi in segs]
    total = sum(int(((hi - lo) / STEP).to_integral_value()) for lo, hi in segs)
    return f"{cid} ({len(segs)} bands; {fmt_dec(min(lows))}–{fmt_dec(max(highs))}) [{total}]"

# ---------- Column plan (order & custom names) ----------
def _normalize_plan(files: List[UploadFile], plan_json: str | None):
    """Return list of column specs: [{'file': UploadFile, 'name': str}], in requested order."""
    n = len(files)
    default_order = list(range(n))
    default_names = [(f.filename or f"Table {i+1}") for i, f in enumerate(files)]

    if not plan_json:
        return [{"file": files[i], "name": default_names[i]} for i in default_order]

    try:
        data = json.loads(plan_json)
    except Exception:
        data = {}

    order = data.get("order", default_order)
    order = [i for i in order if isinstance(i, int) and 0 <= i < n]
    seen = set()
    order = [i for i in order if (i not in seen and not seen.add(i))]
    if not order:
        order = default_order

    names = data.get("names", default_names)
    norm_names = []
    for pos, fi in enumerate(order):
        try:
            name = names[fi] if isinstance(names, list) and len(names) > fi else default_names[fi]
        except Exception:
            name = default_names[fi]
        name = str(name).strip() or default_names[fi]
        norm_names.append(name)

    return [{"file": files[fi], "name": norm_names[pos]} for pos, fi in enumerate(order)]

# ---------- Core assembly ----------
def assemble_result(colspecs: List[dict]) -> tuple[pd.DataFrame, dict]:
    """
    colspecs: [{'file': UploadFile, 'name': 'Column Name'}, ...] — already in desired order.
    Returns (df, color_cols), where color_cols[col] is per-row color tag: FREE | BLUE | YELLOW.
    """
    grid = build_grid()
    result = grid[["Lower Edge Freq (THz)", "Center Freq (THz)", "Higher Edge Freq (THz)"]].copy()
    color_cols = {}

    for spec in colspecs:
        f = spec["file"]
        colname = spec["name"]

        raw = f.file.read()
        df = read_custom_tsv_bytes(raw)
        req = extract_required(df)

        # parse passbands (Decimal)
        req["segments"] = req["passband list"].map(parse_passband_list)
        req["base_label"] = req.apply(lambda r: base_label_for_segments(r["Circuit ID"], r["segments"]), axis=1)

        # derive OCG extras (only for FS OCG; FS NONE adds nothing)
        extras = []
        for _, r in req.iterrows():
            extras.extend(derive_extra_segments(r["Circuit ID"], r.get("Freq Slot Plan Type", ""), r["segments"]))

        texts = []
        colors = []
        for _, slot in grid.iterrows():
            l, u = slot["_lo"], slot["_hi"]   # Decimal; interval is [l, u)
            labs, cols = [], []

            # base overlaps
            for _, r in req.iterrows():
                for lo, hi in r["segments"]:
                    # overlap of [lo,hi) with [l,u) ⇔ hi > l and lo < u
                    if (hi > l) and (lo < u):
                        labs.append(r["base_label"])
                        cols.append("BLUE")
                        break

            # extra OCG slices
            for ex in extras:
                lo, hi = ex["lo"], ex["hi"]
                if (hi > l) and (lo < u):
                    labs.append(ex["label"])
                    cols.append("YELLOW")

            if labs:
                uniq, seen, ucols = [], set(), []
                for lab, ctag in zip(labs, cols):
                    if lab not in seen:
                        uniq.append(lab); ucols.append(ctag); seen.add(lab)
                texts.append(", ".join(uniq))
                colors.append("YELLOW" if "YELLOW" in ucols else "BLUE")
            else:
                texts.append("FREE")
                colors.append("FREE")

        result[colname] = texts
        color_cols[colname] = colors

    return result, color_cols

# ---------- Excel writer with merges & colors ----------
def to_excel_merged_colored(df: pd.DataFrame, color_cols: dict) -> bytes:
    import xlsxwriter
    bio = io.BytesIO()
    wb = xlsxwriter.Workbook(bio, {"in_memory": True})
    ws = wb.add_worksheet("result")

    # formats
    hdr_fmt = wb.add_format({"bold": True, "bg_color": "#FFC000", "align": "center", "valign": "vcenter", "border": 1})
    grid_hdr_fmt = wb.add_format({"bold": True, "bg_color": "#FFE699", "align": "center", "valign": "vcenter", "border": 1})
    fmt_blue   = wb.add_format({"bg_color": "#CCECFF", "valign": "vcenter", "border": 1})
    fmt_green  = wb.add_format({"bg_color": "#C6EFCE", "valign": "vcenter", "border": 1})
    fmt_yellow = wb.add_format({"bg_color": "#FFF2CC", "valign": "vcenter", "border": 1})
    grid_fmt   = wb.add_format({"align": "center", "valign": "vcenter", "border": 1, "num_format": "0.######"})

    # headers
    for c, name in enumerate(df.columns):
        ws.write(0, c, name, grid_hdr_fmt if c < 3 else hdr_fmt)
    ws.set_column(0, 2, 16)
    ws.set_column(3, df.shape[1]-1, 86)

    nrows = df.shape[0]

    # frequency grid numbers
    for r in range(1, nrows+1):
        for c in range(0, 3):
            ws.write_number(r, c, float(df.iat[r-1, c]), grid_fmt)

    # merges per column using (text, color) runs
    def apply_merges(col_idx: int, col_name: str):
        texts = df[col_name].tolist()
        colors = color_cols[col_name]
        r = 1
        while r <= nrows:
            val = texts[r-1]
            tag = colors[r-1]
            run_start = r
            run_end = r
            while run_end + 1 <= nrows and texts[run_end] == val and colors[run_end] == tag:
                run_end += 1
            fmt = fmt_green if tag == "FREE" else (fmt_yellow if tag == "YELLOW" else fmt_blue)
            if run_end > run_start:
                ws.merge_range(run_start, col_idx, run_end, col_idx, val, fmt)
            else:
                ws.write(run_start, col_idx, val, fmt)
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
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<h1>Static index not found</h1>", status_code=500)

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/preview", response_class=HTMLResponse)
async def preview(
    files: List[UploadFile] = File(...),
    start: float | None = None,
    rows: int = 150,
    plan: str | None = Form(None),   # order & names mapping from UI
):
    """
    HTML preview of the result. Optional window:
      - start: starting frequency in THz (query param)
      - rows : number of rows (default 150, max 1500, query param)
      - plan : JSON mapping with desired order & custom column names (form field)
    """
    colspecs = _normalize_plan(files, plan)
    df, color_cols = assemble_result(colspecs)
    rows = max(1, min(1500, int(rows)))
    if start is not None:
        try:
            base = float(GRID_LOW)
            step = float(STEP)
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
    plan: str | None = Form(None),   # order & names mapping from UI
):
    colspecs = _normalize_plan(files, plan)
    df, color_cols = assemble_result(colspecs)
    xls = to_excel_merged_colored(df, color_cols)

    # Имя вида: optical_circuits_merge_1407_19092025.xlsx (Asia/Yerevan)
    now = datetime.now(ZoneInfo("Asia/Yerevan"))
    fname = f"optical_circuits_merge_{now:%H%M}_{now:%d%m%Y}.xlsx"

    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(
        io.BytesIO(xls),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )
