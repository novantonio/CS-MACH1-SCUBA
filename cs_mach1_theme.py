"""
cs_mach1_theme.py
-----------------
CS-MACH1 branding & Streamlit page-setup helpers.
Import this module at the top of any CS-MACH1 Streamlit app:

    from cs_mach1_theme import apply_cs_mach1_theme, cs_mach1_footer
"""

import streamlit as st


# ── Palette ───────────────────────────────────────────────────────────────────
BRAND_BLUE   = "#00A6D6"
BRAND_HOVER  = "#007EA3"
TEXT_MUTED   = "#555555"


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = f"""
<style>

/* ── Header ──────────────────────────────────────────────── */
.cs-main-header {{
    font-size: 34px;
    font-weight: 700;
    color: {BRAND_BLUE};
    margin-bottom: 0px;
}}

.cs-sub-header {{
    font-size: 16px;
    color: {TEXT_MUTED};
    margin-bottom: 20px;
}}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton>button {{
    background-color: {BRAND_BLUE};
    color: white;
    border-radius: 8px;
    border: none;
}}

.stButton>button:hover {{
    background-color: {BRAND_HOVER};
    color: white;
}}

/* ── Footer ──────────────────────────────────────────────── */
.cs-footer {{
    text-align: center;
    color: grey;
    font-size: 13px;
    margin-top: 2rem;
}}

</style>
"""


# ── Public helpers ────────────────────────────────────────────────────────────

def apply_cs_mach1_theme(
    page_title: str = "CS-MACH1",
    page_icon: str = "logo.png",
    main_title: str = "🌊 CS-MACH1",
    subtitle: str = "Ocean temperature monitoring platform",
    logo_path: str = "logo.png",
    logo_width: int = 250,
    layout: str = "wide",
) -> None:
    """
    Call once at the top of your Streamlit script (before any other st.* call).
    Sets page config, injects brand CSS, renders the logo and page header.

    Parameters
    ----------
    page_title  : Browser tab title.
    page_icon   : Emoji or path to favicon image.
    main_title  : Large heading shown in the app.
    subtitle    : Smaller grey text shown below the heading.
    logo_path   : Path (or URL) to the logo image.
    logo_width  : Logo display width in pixels.
    layout      : Streamlit page layout ("wide" or "centered").
    """
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
    )

    # Inject brand CSS
    st.markdown(_CSS, unsafe_allow_html=True)

    # Logo
    try:
        st.image(logo_path, width=logo_width)
    except Exception:
        pass  # silently skip if logo not found

    # Heading
    st.markdown(
        f"<div class='cs-main-header'>{main_title}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='cs-sub-header'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def cs_mach1_footer(text: str = "CS-MACH1 Project • Ocean Temperature Monitoring Platform") -> None:
    """Render the standard CS-MACH1 horizontal-rule + footer."""
    st.markdown("---")
    st.markdown(
        f"<div class='cs-footer'>{text}</div>",
        unsafe_allow_html=True,
    )
