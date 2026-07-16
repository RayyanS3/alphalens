from __future__ import annotations

import logging
import requests
from edgar import set_identity, Company

from src.cache import cached
from src.config import (
    SEC_USER_AGENT,
    SEC_TICKERS_URL,
    SEC_SUBMISSIONS_URL,
    USEFUL_FORMS,
    FILINGS_LIMIT,
    FILING_MAX_CHARS,
    CACHE_TTL_FILINGS,
)

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 15
_SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
set_identity(SEC_USER_AGENT)


# @cached(ttl=CACHE_TTL_FILINGS)
# def get_cik(ticker: str) -> str:
#     try:
#         response = requests.get(SEC_TICKERS_URL, headers=_SEC_HEADERS, timeout=_HTTP_TIMEOUT)
#     except requests.RequestException as e:
#         raise ValueError(f"SEC ticker lookup request failed: {e}") from e

#     if response.status_code != 200:
#         raise ValueError(f"SEC ticker lookup failed: status {response.status_code}")

#     mapping = response.json()
#     for entry in mapping.values():
#         if entry.get("ticker", "").upper() == ticker.upper():
#             return str(entry["cik_str"]).zfill(10)

#     raise ValueError(f"No CIK found for ticker '{ticker}'.")


# @cached(ttl=CACHE_TTL_FILINGS)
# def get_filings(ticker: str, limit: int = FILINGS_LIMIT) -> list[dict]:
#     cik = get_cik(ticker)

#     try:
#         response = requests.get(
#             SEC_SUBMISSIONS_URL.format(cik=cik),
#             headers=_SEC_HEADERS,
#             timeout=_HTTP_TIMEOUT,
#         )
#     except requests.RequestException as e:
#         raise ValueError(f"SEC filings request failed for {ticker}: {e}") from e

#     if response.status_code != 200:
#         raise ValueError(f"SEC filings lookup failed: status {response.status_code}")

#     recent = response.json()["filings"]["recent"]
#     forms = recent["form"]
#     dates = recent["filingDate"]
#     accessions = recent["accessionNumber"]
#     primary_docs = recent["primaryDocument"]

#     filings: list[dict] = []
#     for i, form in enumerate(forms):
#         if form not in USEFUL_FORMS:
#             continue

#         accession_clean = accessions[i].replace("-", "")
#         doc_url = _ARCHIVE_URL.format(
#             cik=int(cik),
#             accession=accession_clean,
#             doc=primary_docs[i],
#         )

#         filings.append({"form": form, "date": dates[i], "url": doc_url})

#         if len(filings) >= limit:
#             break

#     logger.info("Found %d filings for %s", len(filings), ticker)
#     return filings


def get_filing_text(ticker: str, form: str = "10-Q", max_chars: int = FILING_MAX_CHARS) -> str:
    company = Company(ticker)
    filing = company.get_filings(form=form).latest(1)
    if filing is None:
        raise ValueError(f"No {form} filing found for {ticker}.")

    text = " ".join(filing.text().split())
    return text[:max_chars]