"""
app.py — Phase 6
Streamlit web UI for the Breast Cancer RAG system.

This file is intentionally kept thin: it only handles user input,
calls the FastAPI backend, and renders the response.  All the heavy
work (embedding, retrieval, Gemini generation) stays in the API process.

Run:
    streamlit run app.py
(The FastAPI server must be running in a separate terminal first.)
"""

import requests
import streamlit as st

# Change API_BASE_URL if you deploy the backend somewhere other than localhost.
API_BASE_URL   = "http://127.0.0.1:8000"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

# Gemini + retrieval typically takes 5–15 s; 60 s is a safe ceiling.
REQUEST_TIMEOUT_SECONDS = 60


st.set_page_config(
    page_title="Breast Cancer Research Assistant",
    page_icon="🔬",
    layout="centered",
)

st.title("🔬 Breast Cancer Research Assistant")
st.markdown(
    """
    Ask any clinical question about breast cancer treatment.
    This assistant searches **~2,500 PubMed abstracts** (2015–2025) and
    generates a grounded answer with inline citations so you can trace
    every claim back to the original paper.

    > **Note:** This is a research prototype, not a substitute for
    > medical advice.
    """
)

st.divider()


# Using st.form so the app only fires when the user explicitly presses Submit,
# not on every keystroke.
with st.form("query_form"):
    question = st.text_area(
        label="Your question",
        placeholder="e.g. What are first-line treatments for HER2-positive breast cancer?",
        height=100,
    )

    # More context can improve answer quality but also increases latency and
    # the chance of Gemini hitting its input token limit.
    k = st.slider(
        label="Number of abstracts to retrieve as context",
        min_value=1,
        max_value=20,
        value=5,
        help="Higher values give Gemini more evidence to draw on, "
             "but also make the request slower.",
    )

    submitted = st.form_submit_button("Search the literature", type="primary")


if submitted:
    question = question.strip()

    if not question:
        st.warning("Please enter a question before searching.")
        st.stop()

    with st.spinner("Searching the literature — this may take 10–20 seconds..."):
        try:
            response = requests.post(
                QUERY_ENDPOINT,
                json={"question": question, "k": k},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "**Could not reach the API — is the server running?**\n\n"
                "Start it in a separate terminal with:\n"
                "```\nuvicorn api.main:app --reload\n```"
            )
            st.stop()

        except requests.exceptions.Timeout:
            st.error(
                f"The API did not respond within {REQUEST_TIMEOUT_SECONDS} seconds. "
                "Try again or reduce the number of retrieved abstracts."
            )
            st.stop()

        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            st.error(f"API returned an error ({exc.response.status_code}): {detail}")
            st.stop()

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    st.subheader("Answer")
    # st.markdown renders the inline [PMID: XXXXXXXX] citations as plain text;
    # the Sources expander below makes each PMID a clickable link.
    st.markdown(data["answer"])

    st.divider()

    sources      = data.get("sources", [])
    source_count = len(sources)

    with st.expander(f"Sources — {source_count} abstract{'s' if source_count != 1 else ''} retrieved"):
        if not sources:
            st.write("No sources were returned.")
        else:
            for i, source in enumerate(sources, start=1):
                pmid    = source.get("pmid", "")
                title   = source.get("title", "Untitled")
                snippet = source.get("snippet", "")

                pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                st.markdown(f"**{i}. {title}**")
                st.markdown(f"[PMID: {pmid}]({pubmed_url})")

                if snippet:
                    # st.caption renders in a smaller, muted font — good for
                    # the abstract snippet which is supporting detail, not
                    # the primary reading target.
                    st.caption(snippet)

                if i < source_count:
                    st.divider()
