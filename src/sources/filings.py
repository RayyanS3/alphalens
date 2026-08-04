"""SEC filings retrieval via EDGAR.

Fetches filings through edgartools and returns them as structured
SourceDocuments — one per filing section — so every chunk carries the
identity of the filing and the specific section it came from.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone

from edgar import set_identity, Company

from src.config import SEC_USER_AGENT, FILING_MAX_CHARS
from src.models import SourceDocument, make_document_id

logger = logging.getLogger(__name__)

set_identity(SEC_USER_AGENT)

# SEC item numbering is legally fixed, so this mapping is stable.
# Item numbers repeat across parts, hence (part, item) keys.
TENQ_SECTIONS: dict[tuple[str, str], str] = {
    ("PART I", "Item 1"): "Financial Statements",
    ("PART I", "Item 2"): "Management's Discussion and Analysis",
    ("PART I", "Item 3"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("PART I", "Item 4"): "Controls and Procedures",
    ("PART II", "Item 1"): "Legal Proceedings",
    ("PART II", "Item 1A"): "Risk Factors",
    ("PART II", "Item 2"): "Unregistered Sales of Equity Securities",
    ("PART II", "Item 3"): "Defaults Upon Senior Securities",
    ("PART II", "Item 4"): "Mine Safety Disclosures",
    ("PART II", "Item 5"): "Other Information",
    ("PART II", "Item 6"): "Exhibits",
}

TENK_SECTIONS: dict[tuple[str, str], str] = {
    ("PART I", "Item 1"): "Business",
    ("PART I", "Item 1A"): "Risk Factors",
    ("PART I", "Item 3"): "Legal Proceedings",
    ("PART II", "Item 7"): "Management's Discussion and Analysis",
    ("PART II", "Item 7A"): "Quantitative and Qualitative Disclosures About Market Risk",
    ("PART II", "Item 8"): "Financial Statements and Supplementary Data",
    ("PART II", "Item 9A"): "Controls and Procedures",
}

# Sections with little analytical value; skipped to avoid noise in retrieval.
SKIP_SECTIONS = {"Exhibits", "Mine Safety Disclosures", "Defaults Upon Senior Securities", "Financial Statements"}

_MIN_SECTION_CHARS = 200


def _to_utc_datetime(filing_date) -> datetime | None:
    """Normalize edgartools' filing_date to an aware UTC datetime."""
    if filing_date is None:
        return None
    if isinstance(filing_date, datetime):
        return filing_date if filing_date.tzinfo else filing_date.replace(tzinfo=timezone.utc)
    return datetime.combine(filing_date, time.min, tzinfo=timezone.utc)


def _section_map(form: str) -> dict[tuple[str, str], str]:
    """Return the (part, item) -> section-name mapping for a form type."""
    return TENK_SECTIONS if form.upper().startswith("10-K") else TENQ_SECTIONS


def get_filing_documents(
    ticker: str,
    form: str = "10-Q",
    max_chars: int = FILING_MAX_CHARS,
) -> list[SourceDocument]:
    """Fetch a company's latest filing as one SourceDocument per section.

    Falls back to a single whole-filing document if section extraction is
    unavailable, so ingestion degrades gracefully rather than failing.

    Args:
        ticker: The stock ticker symbol.
        form: The filing form type (e.g. "10-Q", "10-K").
        max_chars: Maximum characters retained per section.

    Returns:
        A list of SourceDocuments, one per extracted section.

    Raises:
        ValueError: If no filing of the requested form exists.
    """
    filing = Company(ticker).get_filings(form=form).latest(1)
    if filing is None:
        raise ValueError(f"No {form} filing found for {ticker}.")

    accession = getattr(filing, "accession_no", None)
    filing_form = getattr(filing, "form", form)
    filed_at = _to_utc_datetime(getattr(filing, "filing_date", None))
    company_name = getattr(filing, "company", ticker)
    url = getattr(filing, "filing_url", None)

    def _build(section_name: str | None, text: str) -> SourceDocument:
        key = f"{accession}:{section_name}" if section_name else str(accession)
        title = f"{company_name} {filing_form}"
        return SourceDocument(
            document_id=make_document_id(ticker, "filing", key),
            ticker=ticker,
            source_type="filing",
            title=f"{title} — {section_name}" if section_name else title,
            text=text[:max_chars],
            url=url,
            published_at=filed_at,
            filing_form=filing_form,
            accession_number=accession,
            section=section_name,
        )

    documents: list[SourceDocument] = []

    try:
        obj = filing.obj()
        for (part, item), section_name in _section_map(filing_form).items():
            if section_name in SKIP_SECTIONS:
                continue
            try:
                raw = obj.get_item_with_part(part, item)
            except Exception:
                continue
            if not raw:
                continue
            text = " ".join(str(raw).split())
            if len(text) >= _MIN_SECTION_CHARS:
                documents.append(_build(section_name, text))
    except Exception as e:
        logger.warning("Section extraction failed for %s %s: %s", ticker, filing_form, e)

    if not documents:
        logger.warning("No sections extracted for %s; falling back to full text.", ticker)
        documents = [_build(None, " ".join(filing.text().split()))]

    logger.info(
        "Fetched %s for %s (accession %s, filed %s): %d sections, %d chars.",
        filing_form, ticker, accession, filed_at.date() if filed_at else "unknown",
        len(documents), sum(len(d.text) for d in documents),
    )
    return documents