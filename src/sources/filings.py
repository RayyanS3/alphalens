from __future__ import annotations
import logging
from edgar import set_identity, Company

from src.cache import cached
from src.config import (
    SEC_USER_AGENT,
    FILING_MAX_CHARS,
)

logger = logging.getLogger(__name__)
set_identity(SEC_USER_AGENT)

def get_filing_text(ticker: str, form: str = "10-Q", max_chars: int = FILING_MAX_CHARS) -> str:
    company = Company(ticker)
    filing = company.get_filings(form=form).latest(1)
    if filing is None:
        raise ValueError(f"No {form} filing found for {ticker}.")

    text = " ".join(filing.text().split())
    return text[:max_chars]