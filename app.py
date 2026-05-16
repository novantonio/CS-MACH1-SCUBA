"""
app.py
------
CS-MACH1 EnvLogger pipeline — Streamlit Cloud single-file app.

Layout
------
For every uploaded CSV:
  • Plot 1 – Temperature time-series (raw + rolling mean)
  • Plot 2 – CORA interannual DOY scatter + THIS logger's markers (mean & median)

After all individual files:
  • Plot 3 – CORA interannual DOY scatter + ALL logger markers (mean & median)
  • Summary table (mean, median, std per file)
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── CS-MACH1 branding ─────────────────────────────────────────────────────────
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

CORA_URL_TEMPLATE = (
    "https://erddap.emodnet-physics.eu/erddap/griddap/"
    "INSITU_GLO_PHY_TS_OA_MY_013_052_TEMP.csv"
    "?TEMP%5B(1990-01-01T00:00:00Z):1:(2023-06-15T00:00:00Z)%5D"
    "%5B(1.0):1:(1)%5D"
    "%5B({lat}):1:({lat})%5D"
    "%5B({lon}):1:({lon})%5D"
)

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


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
        latitude  = df.iloc[15, 1]
        longitude = df.iloc[16, 1]
    else:
        latitude  = df.iloc[16, 1]
        longitude = df.iloc[17, 1]

    latitude  = pd.to_numeric(latitude,  errors="coerce")
    longitude = pd.to_numeric(longitude, errors="coerce")

    if pd.isna(latitude) or pd.isna(longitude):
        latitude  = DEFAULT_LATITUDE
        longitude = DEFAULT_LONGITUDE

    return LoggerMetadata(
        serial=serial,
        custom_name=custom_name,
        sampling_frequency=sampling_frequency,
        latitude=latitude,
        longitude=longitude,
    )


def parse_envlog_csv(df: pd.DataFrame) -> pd.DataFrame:
    metadata = extract_metadata(df)

    clean_df = (
        df.iloc[21:, :]
        .dropna()
        .reset_index(drop=True)
    )
    clean_df.columns = ["time", "temperature"]
    clean_df["time"]        = pd.to_datetime(clean_df["time"],       errors="coerce")
    clean_df["temperature"] = pd.to_numeric(clean_df["temperature"], errors="coerce")

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
    """Append mean and median columns (computed on rolling mean if available)."""
    result = df.copy()
    src = (
        result["temperature_rolling_mean"]
        if "temperature_rolling_mean" in result.columns
        else result["temperature"]
    )
    result["temperature_mean"]   = src.mean()
    result["temperature_median"] = src.median()
    return result


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


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_series_and_doy(
    sdata: pd.DataFrame,
    cora_df: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """
    Combined 1-row × 2-column figure per logger file:
      Left  (ax1) – Temperature time-series (raw + rolling mean)
      Right (ax2) – CORA interannual DOY scatter + mean & median markers
                    (no legend on ax2; markers annotated directly)

    Mean marker  : crimson  (filled)
    Median marker: darkorange (filled)
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(18, 5),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )

    label  = sdata["custom_name"].iloc[0]
    yr     = sdata["time"].iloc[0].year

    # ── LEFT: time-series ─────────────────────────────────────────────────────
    ax1.plot(
        sdata["time"], sdata["temperature"],
        alpha=0.4, linewidth=0.8,
        color="steelblue", label="Raw temperature",
    )
    if "temperature_rolling_mean" in sdata.columns:
        ax1.plot(
            sdata["time"], sdata["temperature_rolling_mean"],
            linewidth=2, color="tomato", label="Rolling mean",
        )

    t_mean = sdata["temperature"].mean()
    t_med  = sdata["temperature"].median()

    ax1.axhline(
        t_mean, color="crimson", linewidth=1.4, linestyle="--",
        label=f"Mean {t_mean:.2f} °C",
    )
    ax1.axhline(
        t_med, color="darkorange", linewidth=1.4, linestyle="--",
        label=f"Median {t_med:.2f} °C",
    )

    ax1.legend(fontsize=8)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title(f"Time Series — {label} ({yr})")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="x", rotation=25)

    # ── RIGHT: DOY vs CORA ────────────────────────────────────────────────────
    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    for colour, (year, year_data) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        doy = year_data["time"].dt.dayofyear
        ax2.plot(doy, year_data["TEMP"],
                 marker=".", markersize=4, linestyle="--",
                 color=colour, alpha=0.6)

    d      = sdata["time"].iloc[0].timetuple().tm_yday
    t_mean = sdata["temperature"].mean()
    t_med  = sdata["temperature"].median()
    marker = _year_marker(yr)

    # Mean — crimson filled
    ax2.plot(
        d, t_mean,
        marker=marker, markersize=22, linestyle="None",
        color="crimson", markeredgecolor="black", markeredgewidth=0.8,
        zorder=5,
    )

    # Median — darkorange filled
    ax2.plot(
        d, t_med,
        marker=marker, markersize=22, linestyle="None",
        color="darkorange", markeredgecolor="black", markeredgewidth=0.8,
        zorder=5,
    )

    # Dotted connector
    ax2.plot(
        [d, d], [t_mean, t_med],
        color="grey", linewidth=1.2, linestyle=":",
        zorder=4,
    )

    # Direct annotations instead of a legend
    offset = (t_mean - t_med) * 0.15  # tiny nudge so labels don't overlap
    ax2.annotate(
        f"mean {t_mean:.2f} °C",
        xy=(d, t_mean), xytext=(d + 4, t_mean + abs(offset) + 0.05),
        fontsize=8, color="crimson", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="crimson", lw=0.8),
    )
    ax2.annotate(
        f"median {t_med:.2f} °C",
        xy=(d, t_med), xytext=(d + 4, t_med - abs(offset) - 0.25),
        fontsize=8, color="darkorange", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="darkorange", lw=0.8),
    )

    ax2.set_xlabel("Day of Year")
    ax2.set_ylabel("Temperature [°C]")
    ax2.set_title(
        f"Interannual Variability at ({latitude:.2f}, {longitude:.2f})"
    )
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_doy_all(
    cora_df: pd.DataFrame,
    logger_dfs: dict[str, pd.DataFrame],
    latitude: float,
    longitude: float,
) -> plt.Figure:
    """
    Plot 3 (summary) – CORA interannual DOY scatter +
    ALL logger markers: mean (filled) and median (open).
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    years   = sorted(cora_df["time"].dt.year.unique())
    colours = cm.tab20(np.linspace(0, 1, len(years)))

    for colour, (year, year_data) in zip(colours, cora_df.groupby(cora_df["time"].dt.year)):
        doy = year_data["time"].dt.dayofyear
        ax.plot(doy, year_data["TEMP"],
                marker=".", markersize=4, linestyle="--",
                color=colour, alpha=0.5, label=str(year))

    star_colours = cm.Set1(np.linspace(0, 1, max(len(logger_dfs), 1)))

    for (fname, sdata), sc in zip(logger_dfs.items(), star_colours):
        d      = sdata["time"].iloc[0].timetuple().tm_yday
        t_mean = sdata["temperature"].mean()
        t_med  = sdata["temperature"].median()
        label  = sdata["custom_name"].iloc[0]
        yr     = sdata["time"].iloc[0].year
        marker = _year_marker(yr)

        # Mean — filled
        ax.plot(
            d, t_mean,
            marker=marker, markersize=18, linestyle="None",
            color=sc, markeredgecolor="black", markeredgewidth=0.8,
            label=f"{label} ({yr}) mean",
        )

        # Median — open, same colour edge
        ax.plot(
            d, t_med,
            marker=marker, markersize=18, linestyle="None",
            color="white", markeredgecolor=sc, markeredgewidth=2,
            label=f"{label} ({yr}) median",
        )

        # Connector
        ax.plot([d, d], [t_mean, t_med],
                color="grey", linewidth=1, linestyle=":")

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(
        f"Interannual Temperature Variability at ({latitude:.2f}, {longitude:.2f})\n"
        "— All loggers — filled = mean  ·  open = median —"
    )
    ax.legend(title="Year / Logger", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("### ⚙️ Settings")

    window_size = st.slider("Rolling window", min_value=1, max_value=20, value=5)

    st.divider()

    # File counter — updates as soon as files are staged by the uploader
    n_files = len(st.session_state.get("uploaded_files", []))
    if n_files > 0:
        st.success(f"📂 {n_files} file{'s' if n_files != 1 else ''} loaded")
    else:
        st.info("📂 No files loaded yet")

    st.divider()

    # Progress bar & status — populated during processing
    sidebar_progress = st.empty()
    sidebar_status   = st.empty()

    st.divider()

    start_button = st.button("▶️ Start Processing", type="primary", use_container_width=True)

    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ── File uploader (main area) ─────────────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload one or more envlog CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.session_state["uploaded_files"] = uploaded_files


# ── Process ───────────────────────────────────────────────────────────────────

if start_button and "uploaded_files" in st.session_state:

    raw_files  = st.session_state["uploaded_files"]
    total      = len(raw_files)
    logger_dfs: dict[str, pd.DataFrame] = {}

    pbar = sidebar_progress.progress(0, text="Starting…")

    for i, file in enumerate(raw_files):
        pct  = int((i / total) * 100)
        text = f"Processing {i + 1}/{total}: {file.name}"
        pbar.progress(pct, text=text)
        sidebar_status.caption(text)

        try:
            raw_df   = pd.read_csv(file)
            clean_df = parse_envlog_csv(raw_df)
            proc_df  = add_rolling_mean(clean_df, window_size=window_size)
            proc_df  = add_temperature_summary(proc_df)
            logger_dfs[file.name] = proc_df
        except Exception as exc:
            st.warning(f"Failed processing **{file.name}**: {exc}")

    pbar.progress(100, text="✅ Done!")
    sidebar_status.caption(
        f"Processed {len(logger_dfs)}/{total} file{'s' if total != 1 else ''} successfully."
    )

    if not logger_dfs:
        st.error("No valid logger datasets found.")
        st.stop()

    st.session_state["logger_dfs"] = logger_dfs


# ── Display ───────────────────────────────────────────────────────────────────

if "logger_dfs" in st.session_state:

    logger_dfs: dict[str, pd.DataFrame] = st.session_state["logger_dfs"]

    # Location from first logger
    first_df  = next(iter(logger_dfs.values()))
    latitude  = float(first_df["latitude"].iloc[0])
    longitude = float(first_df["longitude"].iloc[0])

    # Fetch CORA once
    with st.spinner("Loading CORA data…"):
        cora_df = fetch_cora_data(latitude, longitude)

    if cora_df is None:
        st.error("CORA data could not be fetched. Check your connection and try again.")
        st.stop()

    # ── Per-file section ──────────────────────────────────────────────────────
    for fname, sdata in logger_dfs.items():

        st.subheader(f"📄 {fname}")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean temperature",   f"{sdata['temperature'].mean():.2f} °C")
        col_b.metric("Median temperature", f"{sdata['temperature'].median():.2f} °C")
        col_c.metric("Std deviation",       f"{sdata['temperature'].std():.2f} °C")

        # Combined figure: time-series (left) + DOY vs CORA (right)
        fig12 = plot_series_and_doy(sdata, cora_df, latitude, longitude)
        st.pyplot(fig12)
        plt.close(fig12)

        st.divider()

    # ── Summary section ───────────────────────────────────────────────────────
    st.header("📊 Summary — All Loggers vs CORA")

    rows = []
    for fname, sdata in logger_dfs.items():
        rows.append({
            "File":        fname,
            "Name":        sdata["custom_name"].iloc[0],
            "Month":       sdata["time"].iloc[0].strftime("%B %Y"),
            "Mean (°C)":   round(sdata["temperature"].mean(),   2),
            "Median (°C)": round(sdata["temperature"].median(), 2),
            "Std (°C)":    round(sdata["temperature"].std(),    2),
            "N samples":   len(sdata),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Plot 3 – DOY vs CORA (all loggers)
    fig3 = plot_doy_all(cora_df, logger_dfs, latitude, longitude)
    st.pyplot(fig3)
    plt.close(fig3)

    st.info(
        "⭐ stars = 2025  |  ▲ triangles = 2026  |  ■ squares = 2027  |  ● circles = other  \n"
        "**Filled marker** = mean  ·  **Open marker** = median"
    )

    cs_mach1_footer()
