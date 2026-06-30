"""
app.py
------
CS-MACH1 EnvLogger SCUBA pipeline — Streamlit Cloud single-file app.

Layout
------
1. Upload CSV → Start Processing (right below uploader)
2. Per file: metadata (small) + map (scroll-locked) + metrics + plots
             QC section + per-file CSV export
3. Summary section: table + plot3 + plot4
4. Save PDF (after all plots)
5. Summary CSV export (aggregated row per file, SeaDataNet convention)
6. Zenodo multi-file upload (all per-file CSVs + summary CSV in one deposition)
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

from beacon_api import *
import os

from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

apply_cs_mach1_theme(
    page_title="CS-MACH1 my envlogger pipeline",
    main_title="🌊 CS-MACH1: What does my envlogger dive data say about Sea Water Temperature? 🌡",
    subtitle="Ocean temperature comparison platform (in-situ loggers vs CORA reanalysis)",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.meta-card-small {
    background:#f0f7ff; border-left:3px solid #1976d2;
    border-radius:5px; padding:6px 10px; margin-bottom:6px;
}
.meta-label-small {
    color:#1976d2; font-size:0.62rem; text-transform:uppercase;
    letter-spacing:.05em; font-weight:600;
}
.meta-value-small { color:#1a2a3a; font-size:0.80rem; font-weight:500; margin-top:1px; }
.qc-box {
    background:#f8fbff; border:1.5px solid #bbd4f0;
    border-radius:8px; padding:14px 18px; margin:14px 0;
}
.badge-good  { background:#e6f4ea; color:#2e7d32; border-radius:4px; padding:2px 8px; font-size:.82rem; font-weight:600; }
.badge-flagged { background:#fdecea; color:#c62828; border-radius:4px; padding:2px 8px; font-size:.82rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── NERC vocabulary ───────────────────────────────────────────────────────────
NERC = {
    "sensor_uri":      "https://vocab.nerc.ac.uk/collection/L22/current/TOOL2238/",
    "depth_reference": "http://vocab.nerc.ac.uk/collection/L11/current/D11/",
    "Temp_C_concept":  "https://vocab.nerc.ac.uk/collection/P01/current/TEMPPR01/",
    "Temp_C_unit":     "https://vocab.nerc.ac.uk/collection/P06/current/UPAA/",
}

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_LATITUDE  = 44.376290
DEFAULT_LONGITUDE = 9.071358
TMAX = 32

CORA_URL_TEMPLATE = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def _year_marker(year: int) -> str:
    return {2025: "*", 2026: "^", 2027: "s"}.get(year, "o")


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass
class LoggerMetadata:
    serial: str
    custom_name: str
    sampling_frequency: str
    latitude: float
    longitude: float


# ── Parser ────────────────────────────────────────────────────────────────────
def extract_metadata(df: pd.DataFrame) -> LoggerMetadata:
    serial            = df.iloc[9, 1]
    custom_name       = df.iloc[10, 1]
    sampling_frequency = df.iloc[13, 1]
    has_latitude = "lat" in str(df.iloc[15, 0]).lower()
    if has_latitude:
        latitude  = df.iloc[15, 1]
        longitude = df.iloc[16, 1]
    else:
        latitude  = df.iloc[16, 1]
        longitude = df.iloc[17, 1]
    latitude  = pd.to_numeric(latitude,  errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")
    if pd.isna(latitude) or pd.isna(longitude):
        latitude, longitude = DEFAULT_LATITUDE, DEFAULT_LONGITUDE
    return LoggerMetadata(
        serial=str(serial), custom_name=str(custom_name),
        sampling_frequency=str(sampling_frequency),
        latitude=float(latitude), longitude=float(longitude),
    )


def parse_envlog_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, LoggerMetadata]:
    meta = extract_metadata(df)
    clean_df = df.iloc[21:, :].dropna().reset_index(drop=True)
    clean_df.columns = ["time", "temperature"]
    clean_df["time"]        = pd.to_datetime(clean_df["time"], errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"], errors="coerce")
    clean_df["serial"]             = meta.serial
    clean_df["custom_name"]        = meta.custom_name
    clean_df["sampling_frequency"] = meta.sampling_frequency
    clean_df["latitude"]           = meta.latitude
    clean_df["longitude"]          = meta.longitude
    return clean_df.dropna(), meta


# ── Processing ────────────────────────────────────────────────────────────────
def add_rolling_mean(df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
    result = df.copy()
    result["temperature_rolling_mean"] = result["temperature"].rolling(window=window_size).mean()
    return result


def add_temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    src = result["temperature_rolling_mean"] if "temperature_rolling_mean" in result.columns else result["temperature"]
    result["temperature_mean"]   = src.mean()
    result["temperature_median"] = src.median()
    return result


# ── CORA API ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading CORA climatology…")
def fetch_cora_data(latitude: float, longitude: float) -> pd.DataFrame | None:
    url = CORA_URL_TEMPLATE.format(lat=round(latitude, 4), lon=round(longitude, 4))
    try:
        r = requests.get(url, verify=False, timeout=60)
        r.raise_for_status()
        if "<html" in r.text.lower():
            raise ValueError("CORA returned HTML instead of CSV.")
        df = pd.read_csv(io.StringIO(r.text), skiprows=[1])
        df["time"] = pd.to_datetime(df["time"])
        df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
        return df.dropna()
    except Exception as exc:
        st.warning(f"Could not fetch CORA data: {exc}")
        return None


def cora_to_monthly(cora_df: pd.DataFrame) -> pd.DataFrame:
    return (
        cora_df.assign(month=cora_df["time"].dt.month)
        .groupby("month")["TEMP"]
        .agg(["min", "max", "mean", "std"])
        .reset_index()
    )


# ── WOD API ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading WOD climatology…")
def get_ranges_from_wod(latitude: float, longitude: float) -> pd.DataFrame:
    from beacon_api import Client
    client = Client("https://beacon-wod.maris.nl")
    lat_min, lat_max = round(latitude, 1) - 0.5, round(latitude, 1) + 0.5
    lon_min, lon_max = round(longitude, 1) - 0.5, round(longitude, 1) + 0.5
    qb = client.query()
    qb.add_select_column("wod_unique_cast")
    qb.add_select_column("Temperature", alias="TEMPERATURE")
    qb.add_select_column("Temperature_WODflag", alias="TEMPERATURE_QC")
    qb.add_select_column("z", alias="DEPTH")
    qb.add_select_column("time", alias="TIME")
    qb.add_select_column("lon", alias="LONGITUDE")
    qb.add_select_column("lat", alias="LATITUDE")
    qb.add_range_filter("TIME", "1970-01-01T00:00:00", "2023-01-01T00:00:00")
    qb.add_is_not_null_filter("TEMPERATURE")
    qb.add_not_equals_filter("TEMPERATURE", -1e+10)
    qb.add_equals_filter("TEMPERATURE_QC", 0.0)
    qb.add_range_filter("DEPTH", 0, 10_000)
    qb.add_range_filter("LONGITUDE", lon_min, lon_max)
    qb.add_range_filter("LATITUDE", lat_min, lat_max)
    df = qb.to_pandas_dataframe()
    df = df.rename(columns={"TIME": "time", "TEMPERATURE": "TEMP"})
    return df[["time", "TEMP"]]


# ── QC ────────────────────────────────────────────────────────────────────────
def compute_qc(
    df: pd.DataFrame,
    cora_monthly: pd.DataFrame,
    method: str,
    std_factor: float,
    custom_min: float | None = None,
    custom_max: float | None = None,
) -> pd.Series:
    """1 = good, 0 = outlier, -1 = CORA unavailable."""
    if method == "custom":
        if custom_min is None or custom_max is None:
            return pd.Series(1, index=df.index)
        return ((df["temperature"] >= custom_min) & (df["temperature"] <= custom_max)).astype(int)
    if cora_monthly.empty:
        return pd.Series(-1, index=df.index)
    idx = cora_monthly.set_index("month")
    months = df["time"].dt.month
    if method == "range":
        lo = months.map(idx["min"])
        hi = months.map(idx["max"])
    else:  # std
        lo = months.map(idx["mean"]) - std_factor * months.map(idx["std"])
        hi = months.map(idx["mean"]) + std_factor * months.map(idx["std"])
    return ((df["temperature"] >= lo) & (df["temperature"] <= hi)).astype(int)


def _summary_qc_for_file(
    sdata: pd.DataFrame,
    res: dict,
    cora_monthly: pd.DataFrame,
) -> int:
    """
    Compute summary-level TEMP_QC for one file.
    SeaDataNet convention: 0 = no QC, 1 = good, 4 = bad.
    Uses per-file QC series if available, otherwise CORA range on mean temp.
    """
    tqc = res.get("temp_qc", pd.Series(dtype=int))
    if len(tqc) > 0:
        if -1 in tqc.values:
            return 0  # CORA unavailable → no QC
        return 1 if (tqc == 1).all() else 4
    # no QC run yet → evaluate mean against CORA range
    if cora_monthly.empty:
        return 0
    month = sdata["time"].iloc[0].month
    row = cora_monthly[cora_monthly["month"] == month]
    if row.empty:
        return 0
    t_mean = sdata["temperature"].mean()
    return 1 if (float(row["min"].iloc[0]) <= t_mean <= float(row["max"].iloc[0])) else 4


# ── CSV builders ──────────────────────────────────────────────────────────────
def build_per_file_csv(
    df: pd.DataFrame,
    meta: LoggerMetadata,
    depth_m: float,
    temp_qc: pd.Series,
    qc_method: str,
    std_factor: float,
    custom_min: float | None = None,
    custom_max: float | None = None,
) -> bytes:
    lines = []
    lines.append(f"# sensor_uri, {NERC['sensor_uri']}")
    lines.append(f"# sensor_name, {meta.custom_name or meta.serial}")
    lines.append(f"# serial_number, {meta.serial}")
    lines.append("# crs, WGS84")
    lines.append("# time_format, ISO8601_UTC")
    lines.append(f"# depth_m, {depth_m}")
    lines.append(f"# depth_reference, {NERC['depth_reference']}")
    lines.append(f"# Temp_C_concept, {NERC['Temp_C_concept']}")
    lines.append(f"# Temp_C_unit, {NERC['Temp_C_unit']}")
    if qc_method == "range":
        lines.append("# QC_method, CORA_min_max_range")
    elif qc_method == "std":
        lines.append(f"# QC_method, CORA_mean_pm_{std_factor}std")
    else:
        lines.append(f"# QC_method, Custom_Absolute_Range [{custom_min},{custom_max}]degC")
    lines.append("# temp_QC_flag, 1=good 0=outlier -1=CORA_unavailable")
    lines.append("# project, CS-MACH1 EU Horizon Europe Grant No. 101214613")
    lines.append("#")
    lines.append("time_utc, latitude, longitude, Temp_C, depth_m, temp_QC")
    for pos, (_, row) in enumerate(df.iterrows()):
        t_str  = row["time"].strftime("%Y-%m-%dT%H:%M:%SZ")
        qc_val = int(temp_qc.iloc[pos]) if pos < len(temp_qc) else -1
        lines.append(", ".join([
            t_str,
            str(round(float(row["latitude"]), 6)),
            str(round(float(row["longitude"]), 6)),
            str(round(float(row["temperature"]), 4)),
            str(depth_m),
            str(qc_val),
        ]))
    return "\n".join(lines).encode("utf-8")


def build_summary_csv(
    logger_dfs: dict[str, pd.DataFrame],
    results: dict,
    cora_monthly: pd.DataFrame,
    qc_method_used: str,
) -> bytes:
    lines = []
    lines.append("# CS-MACH1 SCUBA EnvLogger — Summary Report")
    lines.append(f"# Temp_C_concept, {NERC['Temp_C_concept']}")
    lines.append(f"# Temp_C_unit, {NERC['Temp_C_unit']}")
    lines.append("# crs, WGS84")
    lines.append("# time_format, ISO8601_UTC")
    lines.append("# temp_QC_flag, SeaDataNet: 0=no_QC 1=good 4=bad")
    lines.append(f"# QC_method, {qc_method_used}")
    lines.append("# description, One aggregated row per uploaded CSV file")
    lines.append("# project, CS-MACH1 EU Horizon Europe Grant No. 101214613")
    lines.append("#")
    lines.append("date_utc, time_mean_utc, latitude, longitude, Temp_C_mean, Temp_C_median, temp_QC")
    for fname, sdata in logger_dfs.items():
        date_str  = sdata["time"].iloc[0].strftime("%Y-%m-%d")
        mean_time = sdata["time"].mean().strftime("%H:%M:%S")
        lat       = round(float(sdata["latitude"].iloc[0]),  6)
        lon       = round(float(sdata["longitude"].iloc[0]), 6)
        t_mean    = round(sdata["temperature"].mean(),   4)
        t_med     = round(sdata["temperature"].median(), 4)
        qc_val    = _summary_qc_for_file(sdata, results.get(fname, {}), cora_monthly)
        lines.append(", ".join([date_str, mean_time, str(lat), str(lon),
                                str(t_mean), str(t_med), str(qc_val)]))
    return "\n".join(lines).encode("utf-8")


# ── Zenodo (multi-file) ───────────────────────────────────────────────────────
def zenodo_upload_multi(
    files_dict: dict[str, bytes],
    title: str, abstract: str, keywords: list[str],
    creators: list[dict], license_id: str, version: str,
    date_str: str, token: str, sandbox: bool,
) -> tuple[bool, str]:
    base = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
    hdrs = {"Authorization": f"Bearer {token}"}
    payload = {
        "metadata": {
            "title": title, "upload_type": "dataset",
            "description": abstract, "creators": creators,
            "keywords": keywords, "access_right": "open",
            "license": license_id, "version": version,
            "language": "eng", "publication_date": date_str,
            "grants": [{"id": "101214613"}],
            "notes": (
                "Funded by the European Union — Horizon Europe, "
                "Grant Agreement No. 101214613 (CS-MACH1). "
                "https://cordis.europa.eu/project/id/101214613"
            ),
            "communities": [{"identifier": "cs-mach1"}],
        }
    }
    try:
        r1 = requests.post(
            f"{base}/deposit/depositions?community=cs-mach1",
            json=payload,
            headers={**hdrs, "Content-Type": "application/json"},
            timeout=30,
        )
        r1.raise_for_status()
        dep = r1.json()
        bucket  = dep["links"]["bucket"]
        rec_url = dep["links"].get(
            "html",
            f"{'https://sandbox.zenodo.org' if sandbox else 'https://zenodo.org'}/deposit/{dep['id']}",
        )
        for fname, data in files_dict.items():
            r2 = requests.put(
                f"{bucket}/{fname}", data=data,
                headers={**hdrs, "Content-Type": "application/octet-stream"},
                timeout=60,
            )
            r2.raise_for_status()
        return True, rec_url
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return False, str(e)


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor("#f8fbff")
    ax.grid(True, alpha=0.35, color="#d0dce8")


def plot_series_and_doy(
    sdata: pd.DataFrame,
    cora_df: pd.DataFrame,
    latitude: float, longitude: float,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.28})
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    label   = sdata["custom_name"].iloc[0]
    yr      = sdata["time"].iloc[0].year
    day_str = sdata["time"].iloc[0].strftime("%Y-%m-%d")
    t_mean  = sdata["temperature"].mean()
    t_med   = sdata["temperature"].median()
    marker  = _year_marker(yr)
    m_month = sdata["time"].iloc[0].month
    d_doy   = sdata["time"].iloc[0].timetuple().tm_yday

    cora_m       = cora_df.copy()
    cora_m["month"] = cora_m["time"].dt.month
    cora_monthly = cora_m.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()
    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    # [0,0] Time-series
    _style_ax(ax1)
    ax1.plot(sdata["time"], sdata["temperature"], alpha=0.4, lw=0.8,
             color="steelblue", label="Raw temperature")
    if "temperature_rolling_mean" in sdata.columns:
        ax1.plot(sdata["time"], sdata["temperature_rolling_mean"],
                 lw=2, color="tomato", label="Rolling mean")
    ax1.axhline(t_mean, color="crimson",    lw=1.4, ls="--", label=f"Mean {t_mean:.2f} °C")
    ax1.axhline(t_med,  color="darkorange", lw=1.4, ls="--", label=f"Median {t_med:.2f} °C")
    ax1.legend(fontsize=8)
    ax1.set_xlabel("Time (HH:MM)")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(f"Time Series — {label} — {day_str}")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax1.tick_params(axis="x", rotation=25)

    # [0,1] CORA monthly mean ± std
    _style_ax(ax2)
    ax2.scatter(cora_monthly["month"], cora_monthly["mean"],
                color="steelblue", zorder=3, label="CORA monthly mean")
    ax2.errorbar(cora_monthly["month"], cora_monthly["mean"],
                 yerr=cora_monthly["std"], fmt="o", color="steelblue",
                 capsize=3, alpha=0.5, label="± std")
    ax2.plot(m_month, t_mean, marker=marker, markersize=12, ls="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} mean {t_mean:.2f} °C")
    ax2.plot(m_month, t_med, marker=marker, markersize=12, ls="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"{label} median {t_med:.2f} °C")
    ax2.plot([m_month, m_month], [t_mean, t_med],
             color="grey", lw=1.2, ls=":", zorder=4)
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax2.set_xlabel("Month"); ax2.set_ylabel("Temperature [°C]")
    ax2.set_ylim(top=TMAX)
    ax2.set_title("CORA Monthly Mean ± Std vs Logger")
    ax2.legend(fontsize=8)

    def _draw_doy(ax):
        for colour, (year, yd) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
            ax.plot(yd["time"].dt.dayofyear, yd["TEMP"],
                    marker=".", ms=4, ls="--", color=colour, alpha=0.6)
        ax.set_xlabel("Day of Year"); ax.set_ylabel("Temperature [°C]")
        ax.grid(True, alpha=0.3)

    # [1,0] DOY mean
    _draw_doy(ax3)
    ax3.plot(d_doy, t_mean, marker=marker, ms=22, ls="None",
             color="crimson", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"mean {t_mean:.2f} °C")
    ax3.annotate(f"mean {t_mean:.2f} °C", xy=(d_doy, t_mean),
                 xytext=(d_doy+4, t_mean+0.3), fontsize=8, color="crimson",
                 fontweight="bold", arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8))
    ax3.set_title(f"DOY — Mean marker | ({latitude:.2f}, {longitude:.2f})")

    # [1,1] DOY median
    _draw_doy(ax4)
    ax4.plot(d_doy, t_med, marker=marker, ms=22, ls="None",
             color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
             zorder=5, label=f"median {t_med:.2f} °C")
    ax4.annotate(f"median {t_med:.2f} °C", xy=(d_doy, t_med),
                 xytext=(d_doy+4, t_med-0.4), fontsize=8, color="darkorange",
                 fontweight="bold", arrowprops=dict(arrowstyle="-", color="darkorange", lw=0.8))
    ax4.set_title(f"DOY — Median marker | ({latitude:.2f}, {longitude:.2f})")

    fig.suptitle(f"{label} ({yr})", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_doy_all(cora_df, logger_dfs, latitude, longitude) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))
    for colour, (year, yd) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        ax.plot(yd["time"].dt.dayofyear, yd["TEMP"],
                marker=".", ms=4, ls="--", color=colour, alpha=0.5, label=str(year))
    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        d      = sdata["time"].iloc[0].timetuple().tm_yday
        t_mean = sdata["temperature"].mean()
        t_med  = sdata["temperature"].median()
        label  = sdata["custom_name"].iloc[0]
        yr     = sdata["time"].iloc[0].year
        marker = _year_marker(yr)
        ax.plot(d, t_mean, marker=marker, ms=12, ls="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8,
                label=f"{label} ({yr}) mean")
        ax.plot(d, t_med, marker=marker, ms=12, ls="None",
                color="white", markeredgecolor=sc, markeredgewidth=2,
                label=f"{label} ({yr}) median")
        ax.plot([d, d], [t_mean, t_med], color="grey", lw=1, ls=":")
    ax.set_xlabel("Day of Year"); ax.set_ylabel("Temperature [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title(f"Interannual Variability at ({latitude:.2f}, {longitude:.2f})\n"
                 "— All loggers — filled=mean · open=median —")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


def plot_doy_all_mean(cora_df, logger_dfs, latitude, longitude) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    cora_tmp = cora_df.copy()
    cora_tmp["month"] = cora_tmp["time"].dt.month
    cm_stats = cora_tmp.groupby("month")["TEMP"].agg(["mean", "std"]).reset_index()
    ax.scatter(cm_stats["month"], cm_stats["mean"], label="Monthly Mean Temperature")
    ax.errorbar(cm_stats["month"], cm_stats["mean"], yerr=cm_stats["std"],
                fmt="o", capsize=3, label="Monthly Standard Deviation")
    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))
    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        month  = sdata["time"].iloc[0].month
        tavg   = sdata["temperature"].mean()
        tavg2  = sdata["temperature"].median()
        label  = sdata["custom_name"].iloc[0]
        year   = sdata["time"].iloc[0].year
        marker = _year_marker(year)
        ax.plot(month, tavg, marker=marker, ms=12, ls="None",
                color=sc, markeredgecolor="black", markeredgewidth=0.8,
                label=f"{label} ({year}) mean")
        ax.plot(month, tavg2, marker=marker, ms=12, ls="None",
                color="white", markeredgecolor=sc, markeredgewidth=2,
                label=f"{label} ({year}) median")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel("Month"); ax.set_ylabel("Temperature [°C]")
    ax.set_ylim(top=TMAX)
    ax.set_title("CORA vs Multiple Logger Monthly Temperature")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    return fig


# ── Map (scroll-locked) ───────────────────────────────────────────────────────
def render_locked_map(latitude: float, longitude: float, popup_label: str, key: str):
    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[latitude, longitude], zoom_start=12,
                       tiles="OpenStreetMap", scrollWheelZoom=False, dragging=True)
        folium.CircleMarker(
            [latitude, longitude], radius=8,
            color="#1565c0", fill=True, fill_color="#1976d2", fill_opacity=0.75,
            popup=folium.Popup(popup_label, max_width=220),
        ).add_to(m)
        st_folium(m, width=320, height=230, returned_objects=[], key=key)
    except ImportError:
        st.map(pd.DataFrame({"lat": [latitude], "lon": [longitude]}))


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
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
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ── Uploader + Start Processing ───────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload one or more envlog CSV files, then press **▶️ Start Processing**",
    type=["csv"], accept_multiple_files=True,
)
if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files

start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)

# ── Processing loop ───────────────────────────────────────────────────────────
if start_button and "uploaded_files" in st.session_state:
    raw_files = st.session_state["uploaded_files"]
    total = len(raw_files)
    results: dict[str, dict] = {}
    pbar = sidebar_progress.progress(0, text="Starting…")
    for i, file in enumerate(raw_files):
        pbar.progress(int(i / total * 100), text=f"Processing {i+1}/{total}: {file.name}")
        sidebar_status.caption(file.name)
        try:
            raw_df   = pd.read_csv(file)
            clean_df, meta = parse_envlog_csv(raw_df)
            proc_df  = add_rolling_mean(clean_df, window_size=window_size)
            proc_df  = add_temperature_summary(proc_df)
            results[file.name] = {
                "df": proc_df, "meta": meta,
                "temp_qc": pd.Series(dtype=int),
                "qc_method": "range", "std_factor": 2.0,
                "custom_min": None, "custom_max": None,
                "depth_m": 5.0,
                "csv_bytes": None, "csv_fname": None,
            }
        except Exception as exc:
            st.warning(f"Failed processing **{file.name}**: {exc}")
    pbar.progress(100, text="✅ Done!")
    sidebar_status.caption(f"Processed {len(results)}/{total} file(s) successfully.")
    if not results:
        st.error("No valid logger datasets found.")
        st.stop()
    st.session_state["results"] = results
    st.session_state["window_size"] = window_size

# ── Display ───────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results: dict[str, dict] = st.session_state["results"]
    win = st.session_state.get("window_size", 5)

    # Location + CORA from first file
    first_df  = results[next(iter(results))]["df"]
    latitude  = float(first_df["latitude"].iloc[0])
    longitude = float(first_df["longitude"].iloc[0])

    with st.spinner("Loading CORA data…"):
        cora_df = fetch_cora_data(latitude, longitude)
    if cora_df is None:
        st.error("CORA data could not be fetched."); st.stop()

    with st.spinner("Loading WOD data…"):
        wod_df = get_ranges_from_wod(latitude, longitude)

    cora_monthly_global = cora_to_monthly(cora_df)

    pdf_buffer = io.BytesIO()
    pdf = PdfPages(pdf_buffer)

    # ── Per-file section ──────────────────────────────────────────────────────
    for fname, res in results.items():
        sdata = res["df"]
        meta  = res["meta"]

        st.markdown("<hr style='border:1px solid #dde6ef;margin:28px 0 18px 0'>", unsafe_allow_html=True)
        st.subheader(f"📄 {fname}")

        # Metadata + map
        col_meta, col_map = st.columns([2, 1])
        with col_meta:
            t0 = sdata["time"].iloc[0].strftime("%Y-%m-%dT%H:%M:%S UTC")
            cards = [
                ("Sensor serial",       meta.serial),
                ("Sensor name",         meta.custom_name),
                ("Sampling frequency",  meta.sampling_frequency),
                ("Recording start",     t0),
                ("Coordinates",         f"{meta.latitude:.6f}°N, {meta.longitude:.6f}°E"),
                ("Total records",       f"{len(sdata):,}"),
            ]
            c1, c2 = st.columns(2)
            for j, (lbl, val) in enumerate(cards):
                with (c1 if j % 2 == 0 else c2):
                    st.markdown(
                        f"<div class='meta-card-small'>"
                        f"<div class='meta-label-small'>{lbl}</div>"
                        f"<div class='meta-value-small'>{val}</div>"
                        f"</div>", unsafe_allow_html=True)
        with col_map:
            render_locked_map(
                meta.latitude, meta.longitude,
                popup_label=f"<b>{meta.custom_name}</b><br>{t0}",
                key=f"map_{fname}",
            )

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Mean temperature",   f"{sdata['temperature'].mean():.2f} °C")
        mc2.metric("Median temperature", f"{sdata['temperature'].median():.2f} °C")
        mc3.metric("Std deviation",      f"{sdata['temperature'].std():.2f} °C")

        # Plots
        fig12 = plot_series_and_doy(sdata, cora_df, latitude, longitude)
        st.pyplot(fig12)
        pdf.savefig(fig12)
        plt.close(fig12)

        # ── QC section ────────────────────────────────────────────────────────
        st.markdown("<div class='qc-box'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Quality Control")

        qc_col1, qc_col2 = st.columns([2, 1])
        with qc_col1:
            qc_method_choice = st.radio(
                "QC Method",
                ["CORA min–max range", "CORA mean ± N·std", "Custom Thresholds"],
                key=f"qc_method_{fname}", horizontal=True,
                help=(
                    "**CORA min–max**: flags values outside historical monthly [min, max].\n\n"
                    "**mean ± N·std**: flags values outside monthly mean ± N std dev.\n\n"
                    "**Custom Thresholds**: absolute global limits."
                ),
            )
        std_factor_val = 2.0
        c_min_val = float(sdata["temperature"].min())
        c_max_val = float(sdata["temperature"].max())
        with qc_col2:
            if "std" in qc_method_choice:
                std_factor_val = st.slider(
                    "Threshold N (std)", 1.0, 10.0, 2.0, 0.5, key=f"std_f_{fname}")
            elif "Custom" in qc_method_choice:
                cc1, cc2 = st.columns(2)
                c_min_val = cc1.number_input("Min Temp (°C)", value=round(c_min_val - 2.0, 1),
                                             step=0.5, key=f"c_min_{fname}")
                c_max_val = cc2.number_input("Max Temp (°C)", value=round(c_max_val + 2.0, 1),
                                             step=0.5, key=f"c_max_{fname}")

        is_disabled = cora_monthly_global.empty and "Custom" not in qc_method_choice
        if cora_monthly_global.empty and "Custom" not in qc_method_choice:
            st.warning("CORA unavailable for this location — select 'Custom Thresholds'.")

        run_qc = st.button("🔍 Run QC", type="primary",
                           key=f"run_qc_{fname}", disabled=is_disabled)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_qc:
            if "range"  in qc_method_choice: mk = "range"
            elif "std"  in qc_method_choice: mk = "std"
            else:                            mk = "custom"
            tqc = compute_qc(sdata, cora_monthly_global, mk, std_factor_val,
                             custom_min=c_min_val, custom_max=c_max_val)
            results[fname]["temp_qc"]    = tqc
            results[fname]["qc_method"]  = mk
            results[fname]["std_factor"] = std_factor_val
            results[fname]["custom_min"] = c_min_val
            results[fname]["custom_max"] = c_max_val
            st.session_state["results"]  = results
            st.rerun()

        # QC badges
        tqc = res.get("temp_qc", pd.Series(dtype=int))
        if len(tqc) > 0 and -1 not in tqc.values:
            n_good = int((tqc == 1).sum()); n_bad = int((tqc == 0).sum())
            pct = n_good / len(tqc) * 100
            st.markdown(
                f'<span class="badge-good">✓ {n_good} good</span>&nbsp;'
                f'<span class="badge-flagged">⚑ {n_bad} flagged</span>&nbsp;'
                f"— pass rate: **{pct:.1f}%**", unsafe_allow_html=True)
            if n_bad > 0:
                with st.expander(f"🔴 Flagged records ({n_bad})"):
                    flagged = sdata[tqc.values == 0].copy()
                    flagged["temp_QC"] = 0
                    st.dataframe(flagged[["time", "temperature", "temp_QC"]],
                                 use_container_width=True)

        # ── Per-file export ───────────────────────────────────────────────────
        st.markdown("#### 💾 Per-file Export")
        exp_c1, exp_c2 = st.columns([1, 1])
        with exp_c1:
            depth_m_val = st.number_input(
                "Dive depth (m)", min_value=0.0, max_value=500.0,
                value=float(res.get("depth_m", 5.0)), step=0.5,
                key=f"depth_{fname}",
            )
            results[fname]["depth_m"] = depth_m_val

        default_fname = f"{(meta.serial or fname.replace('.csv','')).replace(' ','_')}_QC.csv"
        with exp_c2:
            csv_fname_out = st.text_input("Output filename", value=default_fname,
                                          key=f"csv_fname_{fname}")

        # Always build CSV (using current QC or empty series)
        csv_bytes_out = build_per_file_csv(
            sdata, meta, depth_m_val, tqc,
            res.get("qc_method", "range"),
            res.get("std_factor", 2.0),
            res.get("custom_min"), res.get("custom_max"),
        )
        results[fname]["csv_bytes"] = csv_bytes_out
        results[fname]["csv_fname"] = csv_fname_out
        st.session_state["results"] = results

        st.download_button(
            "📥 Download per-file QC CSV",
            data=csv_bytes_out,
            file_name=csv_fname_out,
            mime="text/csv",
            key=f"dl_{fname}",
        )

    # ── Summary section ───────────────────────────────────────────────────────
    st.header("📊 Summary — All Loggers vs CORA")

    rows = []
    for fname, res in results.items():
        sdata = res["df"]
        rows.append({
            "File":       fname,
            "Month":      sdata["time"].iloc[0].strftime("%B %Y"),
            "Mean (°C)":  round(sdata["temperature"].mean(),   2),
            "Median (°C)":round(sdata["temperature"].median(), 2),
            "Std (°C)":   round(sdata["temperature"].std(),    2),
            "N samples":  len(sdata),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    fig3 = plot_doy_all(cora_df, {f: r["df"] for f, r in results.items()}, latitude, longitude)
    st.pyplot(fig3); pdf.savefig(fig3); plt.close(fig3)

    st.divider()

    fig4 = plot_doy_all_mean(cora_df, {f: r["df"] for f, r in results.items()}, latitude, longitude)
    st.pyplot(fig4); pdf.savefig(fig4); plt.close(fig4)

    st.info(
        "⭐ stars = 2025 | ▲ triangles = 2026 | ■ squares = 2027 | ● circles = other\n"
        "**Filled marker** = mean · **Open marker** = median"
    )

    st.divider()

    # ── PDF download ──────────────────────────────────────────────────────────
    pdf.close()
    pdf_buffer.seek(0)
    st.download_button(
        "💾 Save PDF",
        data=pdf_buffer,
        file_name="cs_mach1_scuba_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.divider()

    # ── Summary CSV export ────────────────────────────────────────────────────
    st.subheader("📋 Summary CSV Export")
    st.caption(
        "One aggregated row per file — date, mean time, lat/lon, mean/median temperature, "
        "TEMP_QC (SeaDataNet: 0=no QC · 1=good · 4=bad)"
    )

    # Determine which QC method was used (majority vote or first)
    qc_methods_used = [r.get("qc_method", "range") for r in results.values()]
    qc_method_summary_label = qc_methods_used[0] if qc_methods_used else "CORA_min_max_range"

    summary_bytes = build_summary_csv(
        {f: r["df"] for f, r in results.items()},
        results,
        cora_monthly_global,
        qc_method_summary_label,
    )
    st.download_button(
        "📥 Download Summary CSV",
        data=summary_bytes,
        file_name="cs_mach1_scuba_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()

    # ── Zenodo multi-file upload ───────────────────────────────────────────────
    st.subheader("📤 Upload All Files to Zenodo")
    st.caption("Uploads all per-file QC CSVs + summary CSV in a single Zenodo deposition.")

    # Collect files to upload
    zen_files: dict[str, bytes] = {}
    for fname, res in results.items():
        cb = res.get("csv_bytes")
        cf = res.get("csv_fname") or fname.replace(".csv", "_QC.csv")
        if cb is not None:
            zen_files[cf] = cb
    zen_files["cs_mach1_scuba_summary.csv"] = summary_bytes

    with st.expander("🗂️ Zenodo Metadata", expanded=False):
        first_sdata = results[next(iter(results))]["df"]
        first_meta  = results[next(iter(results))]["meta"]
        dive_date   = first_sdata["time"].iloc[0].strftime("%Y-%m-%d")
        site_name   = first_meta.custom_name or first_meta.serial or "Unknown"

        default_abstract = (
            f"Dataset containing quality-controlled (QC) temperature time series "
            f"recorded by CS-MACH1 loggers at {latitude:.4f}°N, {longitude:.4f}°E, "
            f"starting from {dive_date}. Includes {len(results)} dive file(s) with "
            f"per-timestamp QC flags and an aggregated summary file. "
            f"CORA climatological reference from EMODnet Physics ERDDAP "
            f"(INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP). "
            f"Produced within CS-MACH1 (EU Horizon Europe, Grant No. 101214613)."
        )

        zc1, zc2 = st.columns(2)
        with zc1:
            z_title    = st.text_input("Title *",
                value=f"CS-MACH1 SCUBA EnvLogger Temperature QC — {site_name} ({dive_date})",
                key="z_title")
            z_abstract = st.text_area("Abstract *", value=default_abstract,
                                      height=160, key="z_abstract")
            z_keywords = st.text_input("Keywords (comma-separated)",
                value="ocean temperature, citizen science, CS-MACH1, quality control, EMODnet, SCUBA",
                key="z_keywords")
            z_license  = st.selectbox("License", ["cc-by-4.0","cc-by-nc-4.0","cc0-1.0"], key="z_license")
            z_version  = st.text_input("Version", value="1.0.0", key="z_version")

        with zc2:
            st.markdown("**Authors**")
            st.caption("Lastname, Firstname · ORCID (optional) · Affiliation")
            ak = "zen_authors_scuba"
            if ak not in st.session_state:
                st.session_state[ak] = [{"name": "", "orcid": "", "affiliation": ""}]
            updated = []
            for ai, auth in enumerate(st.session_state[ak]):
                a1, a2, a3, a4 = st.columns([3, 2, 3, 0.6])
                n  = a1.text_input("Name *",    auth["name"],        key=f"{ak}_n{ai}", placeholder="Smith, John")
                o  = a2.text_input("ORCID",     auth["orcid"],       key=f"{ak}_o{ai}", placeholder="0000-0002-…")
                af = a3.text_input("Affiliation",auth["affiliation"], key=f"{ak}_a{ai}", placeholder="Research Center")
                if ai > 0 and a4.button("✕", key=f"{ak}_del{ai}"):
                    st.session_state[ak].pop(ai); st.rerun()
                updated.append({"name": n, "orcid": o, "affiliation": af})
            st.session_state[ak] = updated
            if st.button("➕ Add Co-Author", key="addauth_scuba"):
                st.session_state[ak].append({"name":"","orcid":"","affiliation":""}); st.rerun()

        with st.expander("💶 Grant Funding (pre-filled)"):
            st.info("**CS-MACH1** — EU Horizon Europe · Grant No. **101214613**")

        st.divider()
        tok_c, sb_c = st.columns([3, 1])
        with tok_c:
            if hasattr(st, "secrets") and "ZENODO_TOKEN" in st.secrets:
                zenodo_token = st.secrets["ZENODO_TOKEN"]
                st.success("🔑 Token loaded from Streamlit secrets.")
            else:
                zenodo_token = st.text_input("Zenodo Access Token *",
                                             type="password", key="z_token")
        with sb_c:
            sandbox = st.checkbox("Sandbox", value=True, key="z_sandbox",
                                  help="Use sandbox.zenodo.org for testing.")

        st.markdown(f"**Files to upload:** {', '.join(zen_files.keys())}")

        if st.button("🚀 Push to Zenodo", type="primary", key="z_submit"):
            kw_list  = [k.strip() for k in z_keywords.split(",") if k.strip()]
            creators = []
            for au in st.session_state[ak]:
                if au["name"].strip():
                    c_entry = {"name": au["name"].strip()}
                    if au["orcid"].strip():       c_entry["orcid"]       = au["orcid"].strip()
                    if au["affiliation"].strip():  c_entry["affiliation"] = au["affiliation"].strip()
                    creators.append(c_entry)
            if not zenodo_token:
                st.error("Please provide a Zenodo Access Token.")
            elif not z_title or not z_abstract:
                st.error("Title and Abstract are required.")
            elif not creators:
                st.error("Please provide at least one author.")
            else:
                with st.spinner("Uploading to Zenodo…"):
                    ok, msg = zenodo_upload_multi(
                        files_dict=zen_files,
                        title=z_title, abstract=z_abstract, keywords=kw_list,
                        creators=creators, license_id=z_license, version=z_version,
                        date_str=dive_date, token=zenodo_token, sandbox=sandbox,
                    )
                if ok:
                    st.success("🎉 Upload completed successfully!")
                    st.markdown(f"🔗 **Deposition:** [{msg}]({msg})")
                else:
                    st.error(f"❌ Zenodo error: {msg}")

    st.divider()
    cs_mach1_footer()
