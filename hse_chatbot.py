"""Example script that fetches HSE UEC TGAR data for a given date.

The endpoint expects a query string parameter `EDDATE` in the format DD/MM/YYYY.
By default, this script uses today's date, but you can override it via --date.
"""

import argparse
import datetime as dt
import logging
import os
import sys
import urllib.parse
from html.parser import HTMLParser
from typing import List

import requests
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://uec.hse.ie/uec/TGAR.php"
MODEL = "UCCIX-Mistral-24B"
TEMP = 0.6
MAX_TOK = 8192
QUESTIONS = [
    "Summarise arrivals and admissions",
    "Number of arrivals today in Cork",
]

_client: OpenAI | None = None


def get_client() -> OpenAI:
    api_key = os.getenv("CLOUDCIX_API_KEY")
    if not api_key:
        raise RuntimeError("CLOUDCIX_API_KEY is required for LLM queries")
    global _client
    if _client is None:
        _client = OpenAI(api_key=api_key, base_url="https://ml-openai.cloudcix.com")
    return _client


def format_tgar_date(value: dt.date) -> str:
    """Return DD/MM/YYYY string expected by the TGAR endpoint."""
    return value.strftime("%d/%m/%Y")


def build_tgar_url(value: dt.date) -> str:
    """Construct the TGAR URL with the encoded date parameter."""
    date_str = format_tgar_date(value)
    query = urllib.parse.urlencode({"EDDATE": date_str})
    return f"{BASE_URL}?{query}"


def fetch_tgar_page(value: dt.date, timeout: int = 30) -> str:
    """Download the TGAR HTML page for the supplied date."""
    url = build_tgar_url(value)
    log.info("Fetching TGAR data for %s", format_tgar_date(value))
    log.info("URL: %s", url)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    log.info("Received %d bytes", len(resp.text))
    return resp.text


class FirstTableParser(HTMLParser):
    """Lightweight HTML table parser for the first <table> element.

    This avoids extra dependencies while still surfacing the tabular payload.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.rows: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell: List[str] = []
        self._parsed_any_table = False

    def handle_starttag(self, tag: str, attrs):
        if self._parsed_any_table:
            return
        if tag == "table":
            self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self._current_row = []
        elif tag in ("td", "th") and self.in_row:
            self.in_cell = True
            self._current_cell = []

    def handle_data(self, data: str):
        if self.in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str):
        if self._parsed_any_table:
            return
        if tag in ("td", "th") and self.in_cell:
            cell_text = "".join(self._current_cell).strip()
            self._current_row.append(cell_text)
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self._current_row:
                self.rows.append(self._current_row)
            self.in_row = False
        elif tag == "table" and self.in_table:
            self.in_table = False
            self._parsed_any_table = True


def parse_first_table(html: str) -> List[List[str]]:
    parser = FirstTableParser()
    parser.feed(html)
    return parser.rows


def preview_rows(rows: List[List[str]], limit: int = 5) -> None:
    if not rows:
        log.warning("No table rows found to preview")
        return
    print("\nPreview of first %d row(s):" % min(limit, len(rows)))
    for idx, row in enumerate(rows[:limit], start=1):
        printable = " | ".join(cell if cell else "-" for cell in row)
        print(f"{idx:02d}: {printable}")


def parse_cli_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must be in ISO format YYYY-MM-DD"
        ) from exc


def rows_to_context(rows: List[List[str]]) -> str:
    if not rows:
        return "No table rows available."
    lines: List[str] = []
    for idx, row in enumerate(rows, start=1):
        printable = " | ".join(cell if cell else "-" for cell in row)
        lines.append(f"Row {idx}: {printable}")
    return "\n".join(lines)


def ask_llm_about_tgar(question: str, rows: List[List[str]], data_date: dt.date) -> str:
    client = get_client()
    context = rows_to_context(rows)
    messages = [
        {
            "role": "user",
            "content": (
                f"TGAR date: {format_tgar_date(data_date)}\n"
                f"Table context:\n{context}\n"
                f"Question: {question}\n"
            ),
        },
    ]

    log.info("Sending question to LLM (model=%s)", MODEL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
    )
    return resp.choices[0].message.content or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch HSE UEC TGAR data for a date")
    parser.add_argument(
        "--date",
        type=parse_cli_date,
        help="Date in YYYY-MM-DD (defaults to today)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="How many rows to preview from the first table",
    )
    args = parser.parse_args(argv)

    target_date = args.date or dt.date.today()

    try:
        html = fetch_tgar_page(target_date)
    except Exception as exc:  # noqa: BLE001 - surface any fetch issue to the user
        log.error("Failed to fetch TGAR page: %s", exc)
        return 1

    rows = parse_first_table(html)
    log.info("Parsed %d row(s) from the first table", len(rows))
    preview_rows(rows, limit=args.limit)

    if not rows:
        log.error("Cannot answer questions because no rows were parsed from the TGAR page")
        return 1

    for q in QUESTIONS:
        try:
            answer = ask_llm_about_tgar(
                question=q,
                rows=rows,
                data_date=target_date,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("LLM query failed for '%s': %s", q, exc)
            return 1

        print(f"\nQuestion: {q}\nAnswer:\n{answer}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
