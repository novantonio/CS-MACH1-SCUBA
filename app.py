"""
app.py
------
CS-MACH1 EnvLogger pipeline — Streamlit Cloud single-file app.

Changes vs CS-MACH1-SCUBA/main:
  1. ax1 now shows BOTH the raw clean_df trace AND the rolling-mean proc_df trace.
  2. Moving the rolling-window slider immediately re-processes all files without
     needing to press "Start Processing" again.
  3. clean_dfs stored separately in session_state so re-processing is free.
  4. plot_doy_all_mean: removed duplicated inner loop; mean vs median use
     distinct colours (crimson / darkorange) and the correct variable (tavg2).
  5. plot_series_and_doy signature extended with clean_df parameter.
  6. Minor fixes: variable shadowing in loops, unused assignments removed.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from datetime import datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from reportlab.lib import colors as rl_colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm as rl_cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

apply_cs_mach1_theme(
    page_title="CS-MACH1 my envlogger pipeline",
    main_title="🌊 CS-MACH1: What does my envlogger dive data say about Sea Water Temperature? 🌡",
    subtitle="Ocean temperature comparison platform (in-situ loggers vs CORA reanalysis)",
)


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_LATITUDE  = 44.376290
DEFAULT_LONGITUDE = 9.071358
TMAX              = 32

CORA_URL_TEMPLATE = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _year_marker(year: int) -> str:
    return {2025: "*", 2026: "^", 2027: "s"}.get(year, "o")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


# ── Parser ────────────────────────────────────────────────────────────────────

def extract_metadata(df: pd.DataFrame) -> LoggerMetadata:
    serial             = df.iloc[9, 1]
    custom_name        = df.iloc[10, 1]
    sampling_frequency = df.iloc[13, 1]

    has_latitude = "lat" in str(df.iloc[15, 0]).lower()
    if has_latitude:
        latitude, longitude = df.iloc[15, 1], df.iloc[16, 1]
    else:
        latitude, longitude = df.iloc[16, 1], df.iloc[17, 1]

    latitude  = pd.to_numeric(latitude,  errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")

    if pd.isna(latitude) or pd.isna(longitude):
        latitude, longitude = DEFAULT_LATITUDE, DEFAULT_LONGITUDE

    return LoggerMetadata(
        serial=serial,
        custom_name=custom_name,
        sampling_frequency=sampling_frequency,
        latitude=latitude,
        longitude=longitude,
    )


def parse_envlog_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return the clean parsed DataFrame (no rolling mean yet)."""
    metadata = extract_metadata(df)

    clean_df = df.iloc[21:, :].dropna().reset_index(drop=True)
    clean_df.columns = ["time", "temperature"]
    clean_df["time"]        = pd.to_datetime(clean_df["time"],        errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"],  errors="coerce")

    clean_df["serial"]             = metadata.serial
    clean_df["custom_name"]        = metadata.custom_name
    clean_df["sampling_frequency"] = metadata.sampling_frequency
    clean_df["latitude"]           = metadata.latitude
    clean_df["longitude"]          = metadata.longitude

    return clean_df.dropna()


# ── Processing ────────────────────────────────────────────────────────────────

def add_rolling_mean(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    result = df.copy()
    result["temperature_rolling_mean"] = (
        result["temperature"].rolling(window=window_size).mean()
    )
    return result


def add_temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Append scalar mean and median (computed on rolling mean when available)."""
    result = df.copy()
    src = (
        result["temperature_rolling_mean"]
        if "temperature_rolling_mean" in result.columns
        else result["temperature"]
    )
    result["temperature_mean"]   = src.mean()
    result["temperature_median"] = src.median()
    return result


def reprocess(clean_dfs: dict[str, pd.DataFrame], window_size: int) -> dict[str, pd.DataFrame]:
    """Re-apply rolling mean + summary to every clean DataFrame."""
    out: dict[str, pd.DataFrame] = {}
    for fname, cdf in clean_dfs.items():
        proc = add_rolling_mean(cdf, window_size=window_size)
        proc = add_temperature_summary(proc)
        out[fname] = proc
    return out


# ── CORA API ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading CORA climatology…")
def fetch_cora_data(latitude: float, longitude: float) -> pd.DataFrame | None:
    url = CORA_URL_TEMPLATE.format(lat=round(latitude, 4), lon=round(longitude, 4))
    try:
        response = requests.get(url, verify=False, timeout=60)
        response.raise_for_status()
        if "<html" in response.text.lower():
            raise ValueError("CORA returned an HTML error page instead of CSV.")
        df = pd.read_csv(io.StringIO(response.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"])
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna()
    except Exception as exc:
        st.warning(f"Could not fetch CORA data: {exc}")
        return None


# ── WOD API ───────────────────────────────────────────────────────────────────

def _get_wod_client():
    try:
        from beacon_api import Client          # noqa: PLC0415
        return Client("https://beacon-wod.maris.nl",
                        proxy_headers={"User-Agent": "my-app/1.0 (antonio.novellino@dedagroup.it)"}
                       )
    except ImportError as exc:
        raise ImportError("Run: pip install beacon-api") from exc


@st.cache_data(show_spinner="Downloading WOD data…")
def get_ranges_from_wod(latitude: float, longitude: float) -> pd.DataFrame | None:
    try:
        client  = _get_wod_client()
        lat_min = round(latitude,  1) - 0.1
        lat_max = round(latitude,  1) + 0.1
        lon_min = round(longitude, 1) - 0.1
        lon_max = round(longitude, 1) + 0.1

        qb = client.query()
        qb.add_select_column("wod_unique_cast")
        qb.add_select_column("Temperature",         alias="TEMPERATURE")
        qb.add_select_column("Temperature_WODflag", alias="TEMPERATURE_QC")
        qb.add_select_column("z",                   alias="DEPTH")
        qb.add_select_column("time",                alias="TIME")
        qb.add_select_column("lon",                 alias="LONGITUDE")
        qb.add_select_column("lat",                 alias="LATITUDE")

        qb.add_range_filter("TIME", "1970-01-01T00:00:00", "2023-01-01T00:00:00")
        qb.add_is_not_null_filter("TEMPERATURE")
        qb.add_not_equals_filter("TEMPERATURE", -1e10)
        qb.add_equals_filter("TEMPERATURE_QC", 0.0)
        qb.add_range_filter("DEPTH",     0, 10_000)
        qb.add_range_filter("LONGITUDE", lon_min, lon_max)
        qb.add_range_filter("LATITUDE",  lat_min, lat_max)

        df = qb.to_pandas_dataframe()
        df = df.rename(columns={"TIME": "time", "TEMPERATURE": "TEMP"})
        return df[["time", "TEMP"]]
    except Exception as exc:
        st.warning(f"Could not fetch WOD data: {exc}")
        return None


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_series_and_doy(
    clean_df: pd.DataFrame,       # ← raw parsed data, no rolling mean
    proc_df:  pd.DataFrame,       # ← rolling mean applied
    cora_df:  pd.DataFrame,
    latitude:  float,
    longitude: float,
) -> plt.Figure:
    """
    2 × 2 figure per logger:

      [0,0] ax1 – raw (clean_df, steelblue α=0.35) + rolling mean (proc_df, tomato)
                  + dashed h-lines: mean (crimson), median (darkorange)
      [0,1] ax2 – CORA monthly mean ± std + logger mean & median markers
      [1,0] ax3 – DOY vs CORA interannual + MEAN marker (crimson)
      [1,1] ax4 – DOY vs CORA interannual + MEDIAN marker (darkorange)
    """
    fig, axes = plt.subplots(
        2, 2,
        figsize=(18, 10),
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    label   = proc_df["custom_name"].iloc[0]
    yr      = proc_df["time"].iloc[0].year
    t_mean  = proc_df["temperature"].mean()
    t_med   = proc_df["temperature"].median()
    marker  = _year_marker(yr)
    m_month = proc_df["time"].iloc[0].month
    d_doy   = proc_df["time"].iloc[0].timetuple().tm_yday

    # CORA pre-computations
    cora_m         = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly   = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()
    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    # ── [0,0] Time-series ─────────────────────────────────────────────────────
    # Raw clean data
    ax1.plot(
        clean_df["time"], clean_df["temperature"],
        alpha=0.35, linewidth=0.7, color="steelblue",
        label="Raw data",
    )
    # Rolling mean from proc_df
    if "temperature_rolling_mean" in proc_df.columns:
        window = int(proc_df["temperature_rolling_mean"].isna().sum() + 1)   # approx
        ax1.plot(
            proc_df["time"], proc_df["temperature_rolling_mean"],
            linewidth=2, color="tomato",
            label=f"Rolling mean (w={window_size})",
        )
    # Mean / median reference lines
    ax1.axhline(t_mean, color="crimson",    linewidth=1.4, linestyle="--",
                label=f"Mean   {t_mean:.2f} °C")
    ax1.axhline(t_med,  color="darkorange", linewidth=1.4, linestyle="--",
                label=f"Median {t_med:.2f} °C")

    ax1.legend(fontsize=8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(f"Time Series — {label} ({yr})")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=25)

    # ── [0,1] CORA monthly mean ± std + logger markers ────────────────────────
    ax2.scatter(cora_monthly["month"], cora_monthly["mean"],
                color="steelblue", zorder=3, label="CORA monthly mean")
    ax2.errorbar(cora_monthly["month"], cora_monthly["mean"],
                 yerr=cora_monthly["std"],
                 fmt="o", color="steelblue", capsize=3, alpha=0.5, label="± std")
    ax2.plot(m_month, t_mean,
             marker=marker, markersize=12, linestyle="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} mean {t_mean:.2f} °C")
    ax2.plot(m_month, t_med,
             marker=marker, markersize=12, linestyle="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} median {t_med:.2f} °C")
    ax2.plot([m_month, m_month], [t_mean, t_med],
             color="grey", linewidth=1.2, linestyle=":", zorder=4)

    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Temperature [°C]")
    ax2.set_ylim(top=TMAX)
    ax2.set_title("CORA Monthly Mean ± Std vs Logger |  ({latitude:.2f}, {longitude:.2f})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── shared CORA DOY background ─────────────────────────────────────────────
    def _draw_cora_doy(ax: plt.Axes) -> None:
        for colour, (year, ydata) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
            doy = ydata["time"].dt.dayofyear
            ax.plot(doy, ydata["TEMP"],
                    marker=".", markersize=4, linestyle="--",
                    color=colour, alpha=0.6)
        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Temperature [°C]")
        ax.grid(True, alpha=0.3)

    # ── [1,0] DOY — MEAN marker (crimson) ─────────────────────────────────────
    _draw_cora_doy(ax3)
    ax3.plot(d_doy, t_mean,
             marker=marker, markersize=12, linestyle="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax3.annotate(f"mean {t_mean:.2f} °C",
                 xy=(d_doy, t_mean), xytext=(d_doy + 4, t_mean + 0.3),
                 fontsize=8, color="crimson", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))
    ax3.plot(d_doy, t_med,
             marker=marker, markersize=12, linestyle="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax3.annotate(f"median {t_med:.2f} °C",
                 xy=(d_doy, t_med), xytext=(d_doy + 4, t_med - 0.4),
                 fontsize=8, color="darkorange", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color="darkorange", lw=0.8))
    ax3.set_title(f"CORA Monthly Mean - DOY vs logger |  ({latitude:.2f}, {longitude:.2f})")

    # ── [1,1] DOY — MEDIAN marker (darkorange) ────────────────────────────────
    #_draw_cora_doy(ax4)
    
    
    

    fig.suptitle(f"{label} ({yr})", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_doy_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """DOY scatter + all loggers (filled=mean, open=median)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    for colour, (year, ydata) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        doy = ydata["time"].dt.dayofyear
        ax.plot(doy, ydata["TEMP"],
                marker=".", markersize=4, linestyle="--",
                color=colour, alpha=0.5, label=str(year))

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))

    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        d      = sdata["time"].iloc[0].timetuple().tm_yday
        t_mean = sdata["temperature"].mean()
        t_med  = sdata["temperature"].median()
        lbl    = sdata["custom_name"].iloc[0]
        yr     = sdata["time"].iloc[0].year
        mk     = _year_marker(yr)

        ax.plot(d, t_mean, marker=mk, markersize=12, linestyle="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8,
                label=f"{lbl} ({yr}) mean")
        ax.plot(d, t_med,  marker=mk, markersize=12, linestyle="None",
                color="white", markeredgecolor=sc, markeredgewidth=2,
                label=f"{lbl} ({yr}) median")
        ax.plot([d, d], [t_mean, t_med],
                color="grey", linewidth=1, linestyle=":")

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title(
        f"Interannual Variability at ({latitude:.2f}, {longitude:.2f})\n"
        "— All loggers — filled = mean  ·  open = median —"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_monthly_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
) -> plt.Figure:
    """CORA monthly mean ± std + all loggers (crimson=mean, darkorange=median)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    cora_m         = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly   = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()

    ax.scatter(cora_monthly["month"], cora_monthly["mean"],
               color="steelblue", zorder=3, label="CORA monthly mean")
    ax.errorbar(cora_monthly["month"], cora_monthly["mean"],
                yerr=cora_monthly["std"],
                fmt="o", color="steelblue", capsize=3, alpha=0.5, label="CORA ± std")

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))

    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        month  = sdata["time"].iloc[0].month
        t_mean = sdata["temperature"].mean()
        t_med  = sdata["temperature"].median()
        lbl    = sdata["custom_name"].iloc[0]
        yr     = sdata["time"].iloc[0].year
        mk     = _year_marker(yr)

        ax.plot(month, t_mean, marker=mk, markersize=12, linestyle="None",
                color="crimson", markeredgecolor="black", markeredgewidth=0.8,
                label=f"{lbl} ({yr}) mean {t_mean:.2f} °C")
        ax.plot(month, t_med,  marker=mk, markersize=12, linestyle="None",
                color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
                label=f"{lbl} ({yr}) median {t_med:.2f} °C")
        ax.plot([month, month], [t_mean, t_med],
                color="grey", linewidth=1.2, linestyle=":", zorder=4)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title("CORA vs All Loggers — Monthly Temperature")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── PDF report builder ────────────────────────────────────────────────────────

def _fig_to_rl_image(fig: plt.Figure, width_cm: float = 24.0) -> RLImage:
    """Render a matplotlib figure to an in-memory PNG and wrap it as a ReportLab Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    width_pt  = width_cm * rl_cm
    # Compute height preserving aspect ratio
    fig_w, fig_h = fig.get_size_inches()
    height_pt = width_pt * (fig_h / fig_w)
    img = RLImage(buf, width=width_pt, height=height_pt)
    return img


def build_report_pdf(
    logger_dfs:  dict[str, pd.DataFrame],
    clean_dfs:   dict[str, pd.DataFrame],
    cora_df:     pd.DataFrame,
    latitude:    float,
    longitude:   float,
    window_size: int,
) -> bytes:
    """
    Build a multi-page PDF report and return its bytes.

    Structure
    ---------
    • Cover page  – title, location, date, summary table
    • Per-file pages – 2×2 matplotlib figure (one page per logger)
    • Summary pages – DOY-all figure + monthly-all figure
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(A4),
        leftMargin=1.5 * rl_cm,
        rightMargin=1.5 * rl_cm,
        topMargin=1.5 * rl_cm,
        bottomMargin=1.5 * rl_cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CS_Title",
        parent=styles["Title"],
        fontSize=20,
        textColor=rl_colors.HexColor("#00A6D6"),
        spaceAfter=6,
    )
    h1_style = ParagraphStyle(
        "CS_H1",
        parent=styles["Heading1"],
        fontSize=13,
        textColor=rl_colors.HexColor("#00A6D6"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "CS_Body",
        parent=styles["Normal"],
        fontSize=9,
        spaceAfter=4,
    )
    centre_style = ParagraphStyle(
        "CS_Centre",
        parent=body_style,
        alignment=TA_CENTER,
        textColor=rl_colors.grey,
    )

    story = []

    # ── Cover page ─────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * rl_cm))
    story.append(Paragraph("🌊 CS-MACH1 — EnvLogger Temperature Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=rl_colors.HexColor("#00A6D6")))
    story.append(Spacer(1, 0.4 * rl_cm))

    meta_lines = [
        f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"<b>Location:</b> {latitude:.4f}° N, {longitude:.4f}° E",
        f"<b>Rolling window:</b> {window_size} samples",
        f"<b>Files processed:</b> {len(logger_dfs)}",
        "<b>Reference climatology:</b> CORA (EMODnet-Physics ERDDAP, 1990–2023)",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, body_style))

    story.append(Spacer(1, 0.6 * rl_cm))
    story.append(Paragraph("Summary Table", h1_style))

    # Summary table data
    headers = ["File", "Month", "Mean (°C)", "Median (°C)", "Std (°C)", "N samples"]
    table_data = [headers]
    for fname, proc_df in logger_dfs.items():
        table_data.append([
            fname,
            proc_df["time"].iloc[0].strftime("%B %Y"),
            f"{proc_df['temperature'].mean():.2f}",
            f"{proc_df['temperature'].median():.2f}",
            f"{proc_df['temperature'].std():.2f}",
            str(len(proc_df)),
        ])

    tbl = Table(table_data, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  rl_colors.HexColor("#00A6D6")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  rl_colors.white),
        ("FONTSIZE",    (0, 0), (-1, 0),  9),
        ("FONTSIZE",    (0, 1), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [rl_colors.HexColor("#F0FAFF"), rl_colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#CCCCCC")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)

    story.append(Spacer(1, 1.2 * rl_cm))
    story.append(Paragraph(
        "CS-MACH1 Project · Ocean Temperature Monitoring Platform",
        centre_style,
    ))

    # ── Per-file pages ─────────────────────────────────────────────────────────
    for fname, proc_df in logger_dfs.items():
        story.append(PageBreak())
        lbl = proc_df["custom_name"].iloc[0]
        yr  = proc_df["time"].iloc[0].year
        story.append(Paragraph(f"📄 {lbl} ({yr}) — {fname}", h1_style))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=rl_colors.HexColor("#00A6D6")))
        story.append(Spacer(1, 0.3 * rl_cm))

        # Re-generate the 2×2 figure (matplotlib figures were closed after display)
        fig = plot_series_and_doy(
            clean_dfs[fname], proc_df, cora_df, latitude, longitude
        )
        story.append(_fig_to_rl_image(fig, width_cm=26.0))
        plt.close(fig)

    # ── Summary pages ──────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("📊 Summary — All Loggers vs CORA (DOY)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=rl_colors.HexColor("#00A6D6")))
    story.append(Spacer(1, 0.3 * rl_cm))

    fig_doy = plot_doy_all(cora_df, logger_dfs, latitude, longitude)
    story.append(_fig_to_rl_image(fig_doy, width_cm=26.0))
    plt.close(fig_doy)

    story.append(PageBreak())
    story.append(Paragraph("📊 Summary — All Loggers vs CORA (Monthly)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=rl_colors.HexColor("#00A6D6")))
    story.append(Spacer(1, 0.3 * rl_cm))

    fig_monthly = plot_monthly_all(cora_df, logger_dfs)
    story.append(_fig_to_rl_image(fig_monthly, width_cm=26.0))
  
    plt.close(fig_monthly)

    story.append(Spacer(1, 0.5 * rl_cm))
    story.append(Paragraph(
        "CS-MACH1 Project · Ocean Temperature Monitoring Platform",
        centre_style,
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # The slider value is captured here; its change automatically triggers
    # the reactive re-processing block below (no button press needed).
    window_size = st.slider("Rolling window", min_value=1, max_value=20, value=5)

    st.divider()

    n_files = len(st.session_state.get("uploaded_files", []))
    if n_files > 0:
        st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} loaded")
    else:
        st.info("📂 No files loaded yet")

    st.divider()

    sidebar_progress = st.empty()
    sidebar_status   = st.empty()

    st.divider()

    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)

    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # PDF download — only available after processing
    st.markdown("### 📥 Export")
    if st.button("📄 Generate PDF Report", use_container_width=True):
      with st.spinner("Building PDF…"):
        try:
          pdf_bytes = build_report_pdf(
            logger_dfs   = st.session_state["logger_dfs"],
            clean_dfs    = st.session_state["clean_dfs"],
            cora_df      = fetch_cora_data(
              float(next(iter(st.session_state["logger_dfs"].values()))["latitude"].iloc[0]),
              float(next(iter(st.session_state["logger_dfs"].values()))["longitude"].iloc[0]),
              ),
            latitude     = float(next(iter(st.session_state["logger_dfs"].values()))["latitude"].iloc[0]),
            longitude    = float(next(iter(st.session_state["logger_dfs"].values()))["longitude"].iloc[0]),
            window_size  = window_size,
          )
          st.session_state["pdf_bytes"] = pdf_bytes
        except Exception as exc:
          st.error(f"PDF generation failed: {exc}")

        if "pdf_bytes" in st.session_state:
            fname_ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="⬇️ Download PDF",
                data=st.session_state["pdf_bytes"],
                file_name=f"CS_MACH1_report_{fname_ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


# ── File uploader ─────────────────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or more envlog CSV files, then press 'Start Processing'",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files


# ── Initial processing (button) ───────────────────────────────────────────────

if start_button and "uploaded_files" in st.session_state:

    raw_files  = st.session_state["uploaded_files"]
    total      = len(raw_files)
    clean_dfs: dict[str, pd.DataFrame] = {}

    pbar = sidebar_progress.progress(0, text="Starting…")

    for i, file in enumerate(raw_files):
        pct  = int((i / total) * 100)
        text = f"Parsing {i + 1}/{total}: {file.name}"
        pbar.progress(pct, text=text)
        sidebar_status.caption(text)
        try:
            raw_df   = pd.read_csv(file)
            clean_df = parse_envlog_csv(raw_df)
            clean_dfs[file.name] = clean_df
        except Exception as exc:
            st.warning(f"Failed parsing **{file.name}**: {exc}")

    pbar.progress(100, text="✅ Done!")
    sidebar_status.caption(
        f"Parsed {len(clean_dfs)}/{total} file{'s' if total != 1 else ''} OK."
    )

    if not clean_dfs:
        st.error("No valid logger datasets found.")
        st.stop()

    st.session_state["clean_dfs"]   = clean_dfs
    st.session_state["last_window"] = window_size
    st.session_state["logger_dfs"]  = reprocess(clean_dfs, window_size)


# ── Reactive re-processing when slider moves ──────────────────────────────────
# This runs every time Streamlit re-executes the script (i.e. on every slider
# interaction) and updates logger_dfs without touching the raw CSV parsing.

if (
    "clean_dfs" in st.session_state
    and st.session_state.get("last_window") != window_size
):
    st.session_state["logger_dfs"]  = reprocess(st.session_state["clean_dfs"], window_size)
    st.session_state["last_window"] = window_size
    sidebar_status.caption(f"Re-processed with window = {window_size}")


# ── Display ───────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state:

    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]
    clean_dfs:  dict[str, pd.DataFrame] = st.session_state["clean_dfs"]

    first_df  = next(iter(logger_dfs.values()))
    latitude  = float(first_df["latitude"].iloc[0])
    longitude = float(first_df["longitude"].iloc[0])

    with st.spinner("Loading CORA data…"):
        cora_df = fetch_cora_data(latitude, longitude)

    if cora_df is None:
        st.error("CORA data could not be fetched.")
        st.stop()

#  next version can include
#    with st.spinner("Loading WOD data…"):
#        wod_df = get_ranges_from_wod(latitude, longitude)

    # ── Per-file ──────────────────────────────────────────────────────────────
    for fname, proc_df in logger_dfs.items():

        st.subheader(f"📄 {fname}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean temperature",   f"{proc_df['temperature'].mean():.2f} °C")
        col_b.metric("Median temperature", f"{proc_df['temperature'].median():.2f} °C")
        col_c.metric("Std deviation",       f"{proc_df['temperature'].std():.2f} °C")

        fig_file = plot_series_and_doy(
            clean_dfs[fname], proc_df, cora_df, latitude, longitude
        )
        st.pyplot(fig_file)
        plt.close(fig_file)

        st.divider()

    # ── Summary ───────────────────────────────────────────────────────────────
    st.header("📊 Summary — All Loggers vs CORA")

    rows = []
    for fname, proc_df in logger_dfs.items():
        rows.append({
            "File":        fname,
            "Month":       proc_df["time"].iloc[0].strftime("%B %Y"),
            "Mean (°C)":   round(proc_df["temperature"].mean(),   2),
            "Median (°C)": round(proc_df["temperature"].median(), 2),
            "Std (°C)":    round(proc_df["temperature"].std(),    2),
            "N samples":   len(proc_df),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    fig_doy = plot_doy_all(cora_df, logger_dfs, latitude, longitude)
    fig_monthly = plot_monthly_all(cora_df, logger_dfs)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(fig_doy)
        plt.close(fig_doy)
    
    with col2:
        st.pyplot(fig_monthly)
        plt.close(fig_monthly)

    st.info(
        "⭐ stars = 2025  |  ▲ triangles = 2026  |  ■ squares = 2027  |  ● circles = other  \n"
        "**Filled / crimson** = mean  ·  **darkorange** = median"
    )
    st.divider()
    cs_mach1_footer()
