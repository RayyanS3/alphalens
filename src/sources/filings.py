"""SEC filings retrieval via EDGAR.

Fetches a company's filings through edgartools and returns them as structured
SourceDocuments carrying filing identity (form, date, accession number, URL)
so any claim derived from a filing can be traced back to it.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone

from edgar import set_identity, Company

from src.config import SEC_USER_AGENT, FILING_MAX_CHARS
from src.models import SourceDocument, make_document_id

logger = logging.getLogger(__name__)

set_identity(SEC_USER_AGENT)


def _to_utc_datetime(filing_date) -> datetime | None:
    """Normalize edgartools' filing_date (a date) to an aware UTC datetime."""
    if filing_date is None:
        return None
    if isinstance(filing_date, datetime):
        return filing_date if filing_date.tzinfo else filing_date.replace(tzinfo=timezone.utc)
    return datetime.combine(filing_date, time.min, tzinfo=timezone.utc)


def get_filing_documents(
    ticker: str,
    form: str = "10-Q",
    max_chars: int = FILING_MAX_CHARS,
) -> list[SourceDocument]:
    """Fetch a company's latest filing of a given form as a SourceDocument.

    Args:
        ticker: The stock ticker symbol.
        form: The filing form type (e.g. "10-Q", "10-K").
        max_chars: Maximum characters of filing text to retain.

    Returns:
        A list containing the latest matching filing, or an empty list if none
        is found. (A list, so callers handle multiple filings uniformly once
        multi-filing ingestion is added.)

    Raises:
        ValueError: If no filing of the requested form exists for the ticker.
    """
    company = Company(ticker)
    filing = company.get_filings(form=form).latest(1)

    if filing is None:
        raise ValueError(f"No {form} filing found for {ticker}.")

    text = " ".join(filing.text().split())[:max_chars]
    accession = getattr(filing, "accession_no", None)
    filing_form = getattr(filing, "form", form)
    filed_at = _to_utc_datetime(getattr(filing, "filing_date", None))
    company_name = getattr(filing, "company", ticker)

    document = SourceDocument(
        document_id=make_document_id(ticker, "filing", accession or f"{form}-latest"),
        ticker=ticker,
        source_type="filing",
        title=f"{company_name} {filing_form}",
        text=text,
        url=getattr(filing, "filing_url", None),
        published_at=filed_at,
        filing_form=filing_form,
        accession_number=accession,
    )

    logger.info(
        "Fetched %s for %s (accession %s, filed %s, %d chars).",
        filing_form, ticker, accession, filed_at.date() if filed_at else "unknown", len(text),
    )
    return [document]