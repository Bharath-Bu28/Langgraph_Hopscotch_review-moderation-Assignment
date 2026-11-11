"""From the Hopscotch project root: python -m streamlit run streamlit_app.py"""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from hopscotch.graph import MODES
from hopscotch.schemas import ReviewResult
from pydantic import ValidationError


def run_samples(mode):
    """Use the installed CLI so it can read its own bundled sample format."""
    with TemporaryDirectory() as run_dir:
        command = [
            sys.executable,
            "-m",
            "hopscotch.cli",
            "--provider",
            "offline",
            "--mode",
            mode,
            "--json",
            "--db",
            str(Path(run_dir) / "reviews.sqlite3"),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "The Hopscotch CLI failed.")
    return [
        ReviewResult.model_validate_json(line)
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def saved_results(path):
    rows = []
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(ReviewResult.model_validate_json(line))
                except ValidationError:
                    raise ValueError(
                        f"Invalid review record on line {number}."
                    ) from None
    return rows


def show_results(rows):
    if not rows:
        st.info("There are no reviews in this result set.")
        return
    published, held, alerts = st.columns(3)
    published.metric("Published", sum(row.status == "published" for row in rows))
    held.metric("Held", sum(row.status == "held" for row in rows))
    alerts.metric("Quality alerts", sum(bool(row.alert) for row in rows))
    st.dataframe(
        [
            {
                "Review": row.review_id,
                "Kind": row.kind,
                "Status": row.status,
                "Alert": bool(row.alert),
                "Tool batches": row.tool_rounds,
                "Rewrites": row.revisions,
            }
            for row in rows
        ],
        hide_index=True,
    )
    index = st.selectbox(
        "Inspect a review", range(len(rows)), format_func=lambda i: rows[i].review_id
    )
    row = rows[index]
    st.subheader(row.review_id)
    st.text(row.text)
    for label, value in [
        ("Answer", row.answer),
        ("Public reply", row.reply),
        ("Reason held", row.hold_reason),
        ("Internal quality alert", row.alert),
    ]:
        if value:
            st.markdown(f"**{label}**")
            st.text(value)
    if row.facts:
        with st.expander("Complaint lookup results"):
            st.json(row.facts)
    st.markdown("**Workflow trace**")
    st.code("\n".join(row.trace), language="text")


st.set_page_config(page_title="Hopscotch review moderation", layout="wide")
st.title("Hopscotch review moderation")
source = st.radio(
    "Results", ["Run offline samples", "View saved JSONL"], horizontal=True
)
if source == "View saved JSONL":
    path = Path("output/checked-results.jsonl")
    st.caption(str(path))
    if path.is_file():
        try:
            show_results(saved_results(path))
        except (OSError, ValueError) as exc:
            st.error(f"Could not read the results. {exc}")
    else:
        st.info(
            "Open Streamlit from the project folder containing output/checked-results.jsonl."
        )
else:
    mode = st.selectbox("Workflow mode", MODES, index=MODES.index("checked"))
    if st.button("Run seven reviews", type="primary"):
        with st.spinner("Processing sample reviews"):
            try:
                st.session_state["completed_run"] = (mode, run_samples(mode))
            except (RuntimeError, ValidationError) as exc:
                st.session_state.pop("completed_run", None)
                st.error(str(exc))
    completed = st.session_state.get("completed_run")
    if completed and completed[0] == mode:
        show_results(completed[1])
