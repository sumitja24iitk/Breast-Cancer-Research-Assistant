"""
generate.py — Phase 4
Calls Google Gemini to produce a grounded, cited answer from retrieved abstracts.

CONCEPTS — read before the code
──────────────────────────────────────────────────────────────────────────────

Why constraining the model to retrieved context reduces hallucination
─────────────────────────────────────────────────────────────────────
Large language models are trained to be helpful and fluent.  When asked a
question they don't have confident knowledge about, they often "confabulate" —
they produce text that sounds plausible and well-structured but is factually
wrong.  In medicine, a hallucinated drug dosage or contraindication is
dangerous, not just embarrassing.

Grounding works by reframing the task.  Instead of asking "what do you know
about HER2 treatment?" — an open recall task — we ask "given THESE specific
passages, what do they say?"  This is a reading-comprehension task, which LLMs
are much more reliable at.  The model can still paraphrase or misread a passage,
but it can no longer invent entirely new facts because all facts must trace back
to the provided text.

Why inline PMID citations matter
─────────────────────────────────
Citations serve two audiences:
  1. The end user can click the PMID, read the original abstract, and verify the
     claim themselves — critical for any clinical decision support tool.
  2. The automated evaluation harness (Phase 8) can parse [PMID: XXXXXXXX] tags
     and check whether the cited abstract actually supports the claim (faithfulness
     metric).  Without citations, you can only measure fluency, not accuracy.

Why "I cannot answer from this literature" is the right fallback
────────────────────────────────────────────────────────────────
If the user asks about, say, lung cancer dosing and our corpus has no relevant
abstracts, the top-k retrieved chunks will be only loosely related.  A model
without a clear refusal instruction will try to answer anyway, cherry-picking
tangential details and presenting them confidently.  Giving the model explicit
permission — and instruction — to say "I don't know from this context" is safer
and more honest.  It also signals to the user that they should broaden their
search or consult a clinician, rather than acting on a fabricated answer.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env into os.environ; safe no-op if .env is absent


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# gemini-2.0-flash: stable, fast, free-tier-compatible.
# Switch to "gemini-2.5-pro" for longer / more complex reasoning tasks.
GEMINI_MODEL = "gemini-2.5-flash"

_api_key = os.getenv("GOOGLE_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found.  Add it to your .env file:\n"
        "  GOOGLE_API_KEY=your_key_here\n"
        "Get a free key at https://aistudio.google.com/app/apikey"
    )

_gemini = genai.Client(api_key=_api_key)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is labelled with its PMID so the model can reference it in
    citations.  The numbering ([1], [2], ...) gives the model a shorthand it
    can use in its reasoning even though the final citation must be the PMID.

    Example output:
        [1] PMID: 38123456 | Trastuzumab plus pertuzumab in HER2-positive...
        Background: HER2-positive breast cancer accounts for...

        [2] PMID: 38001234 | Endocrine therapy resistance mechanisms...
        ...
    """
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[{i}] PMID: {chunk['pmid']} | {chunk['title']}\n{chunk['text']}"
        )
    return "\n\n".join(sections)


def _build_prompt(question: str, context_block: str) -> str:
    """
    Assemble the full prompt sent to Gemini.

    Design principles:
    - Role framing ("clinical evidence assistant") nudges the model toward a
      careful, factual tone rather than conversational fluency.
    - Explicit citation format ([PMID: XXXXXXXX]) is machine-parseable for
      the Phase 8 evaluation harness.
    - The refusal instruction is stated as a positive obligation ("respond
      with exactly: ...") not just a prohibition, which is more reliable.
    - "Do NOT draw on knowledge outside the provided abstracts" closes the
      loophole where the model adds plausible-but-uncited background facts.
    """
    return f"""\
You are a clinical evidence assistant specialising in breast cancer treatment.

Your task:
1. Answer the question below using ONLY the research abstracts provided in the
   CONTEXT section.
2. After each factual claim, cite the supporting abstract(s) using this exact
   format: [PMID: XXXXXXXX]
   If multiple abstracts support the same claim, list all: [PMID: 11111111, PMID: 22222222]
3. If the provided abstracts do not contain enough information to answer the
   question, respond with exactly:
   "The available literature does not contain sufficient information to answer this question."
4. Do NOT draw on any knowledge outside the provided abstracts.  Every claim
   must be traceable to a PMID in the context.

QUESTION:
{question}

CONTEXT:
{context_block}

ANSWER:"""


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Generate a grounded, cited answer from Gemini given a question and chunks.

    Parameters
    ----------
    question : str
        The user's clinical question.
    retrieved_chunks : list[dict]
        Output of dense_retrieve() — each dict has pmid, title, text, score.

    Returns
    -------
    str
        Gemini's answer with inline [PMID: XXXXXXXX] citations, or the
        standard refusal string if the context is insufficient.
    """
    if not retrieved_chunks:
        return "The available literature does not contain sufficient information to answer this question."

    context_block = _build_context_block(retrieved_chunks)
    prompt        = _build_prompt(question, context_block)

    response = _gemini.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text.strip()
