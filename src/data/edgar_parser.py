"""
SEC EDGAR Parser for CLO Holdings from N-PORT Filings.

This module provides functionality to:
- Search for N-PORT filings on SEC EDGAR
- Download and cache XML filings
- Parse CLO and loan positions from filings
- Calculate ownership concentration metrics

SEC EDGAR API compliance:
- User-Agent header with contact info required
- Rate limiting to max 10 requests/second
"""

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single CLO/loan position in a fund's portfolio."""

    asset_name: str
    cusip: Optional[str]
    lei: Optional[str]
    asset_category: str
    market_value: float
    percentage_of_portfolio: float
    units: Optional[float] = None
    currency: str = "USD"
    fair_value_level: Optional[str] = None
    issuer_name: Optional[str] = None


@dataclass
class Filing:
    """Represents a parsed N-PORT filing."""

    fund_name: str
    reporting_date: str
    accession_number: str
    cik: str
    total_assets: float
    positions: list[Position] = field(default_factory=list)
    filing_date: Optional[str] = None


@dataclass
class OwnershipMetrics:
    """Ownership concentration metrics for a set of positions."""

    total_clo_positions: int
    total_market_value: float
    average_position_size: float
    top_5_concentration: float  # % of CLO portfolio in top 5 positions
    herfindahl_index: float  # Concentration measure (0-1)
    largest_position_pct: float
    positions_above_5pct: int


class EDGARParser:
    """
    Parser for SEC EDGAR N-PORT filings to extract CLO holdings.

    Handles SEC API compliance including User-Agent headers and rate limiting.
    Downloaded filings are cached locally to minimize API calls.
    """

    BASE_URL = "https://www.sec.gov"
    EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"
    FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"

    # SEC requests max 10 requests/second
    MIN_REQUEST_INTERVAL = 0.1

    # N-PORT XML namespaces
    NPORT_NS = {
        "nport": "http://www.sec.gov/edgar/nport",
        "nportx": "http://www.sec.gov/edgar/nportfiling",
    }

    def __init__(
        self,
        user_agent: str = "LoanLiquidityPredictor research@example.com",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize the EDGAR parser.

        Args:
            user_agent: Required User-Agent header for SEC compliance.
                        Should include application name and contact email.
            cache_dir: Directory for caching downloaded filings.
                       Defaults to data/cache/edgar/ relative to project root.
        """
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json, application/xml, text/html, */*",
            }
        )

        # Set up cache directory
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "data" / "cache" / "edgar"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce SEC rate limiting (max 10 requests/second)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _make_request(
        self, url: str, params: Optional[dict] = None
    ) -> requests.Response:
        """
        Make a rate-limited request to SEC EDGAR.

        Args:
            url: The URL to request
            params: Optional query parameters

        Returns:
            Response object

        Raises:
            requests.RequestException: On request failure
        """
        self._rate_limit()
        logger.debug(f"Requesting: {url}")

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response

    def search_nport_filings(
        self,
        cik: Optional[str] = None,
        form_type: str = "NPORT-P",
        start_date: Optional[str] = None,
        count: int = 10,
    ) -> list[dict]:
        """
        Search for N-PORT filings on SEC EDGAR.

        Args:
            cik: Optional CIK number to filter by specific fund
            form_type: Form type to search for (default: NPORT-P)
            start_date: Start date for search (YYYY-MM-DD format)
            count: Maximum number of results to return

        Returns:
            List of filing metadata dictionaries containing:
            - accession_number
            - cik
            - company_name
            - filed_date
            - form_type
        """
        # Use SEC's full-text search API
        search_url = "https://efts.sec.gov/LATEST/search-index"

        # Build query
        query_parts = [f'formType:"{form_type}"']
        if cik:
            # Normalize CIK to 10 digits with leading zeros
            cik_normalized = cik.lstrip("0").zfill(10)
            query_parts.append(f"ciks:{cik_normalized}")
        if start_date:
            query_parts.append(f'filedAt:[{start_date} TO *]')

        params = {
            "q": " AND ".join(query_parts),
            "dateRange": "custom",
            "startdt": start_date or "2019-01-01",
            "enddt": datetime.now().strftime("%Y-%m-%d"),
            "forms": form_type,
            "from": 0,
            "size": count,
        }

        # Try the newer SEC EDGAR search API
        try:
            search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
            params = {
                "action": "getcompany",
                "type": form_type,
                "dateb": "",
                "owner": "include",
                "count": count,
                "output": "atom",
            }
            if cik:
                params["CIK"] = cik

            response = self._make_request(search_url, params)
            return self._parse_atom_feed(response.text, count)
        except requests.RequestException as e:
            logger.warning(f"Atom feed search failed: {e}, trying alternative API")

        # Fallback: Use company submissions API for specific CIK
        if cik:
            return self._search_by_cik(cik, form_type, count)

        logger.error("Search requires either CIK or working EFTS API")
        return []

    def _parse_atom_feed(self, atom_xml: str, max_results: int) -> list[dict]:
        """Parse SEC EDGAR Atom feed results."""
        results = []
        try:
            root = ET.fromstring(atom_xml)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            for entry in root.findall(".//atom:entry", ns)[:max_results]:
                title = entry.find("atom:title", ns)
                link = entry.find("atom:link", ns)
                updated = entry.find("atom:updated", ns)
                summary = entry.find("atom:summary", ns)

                if title is not None and link is not None:
                    href = link.get("href", "")
                    # Extract accession number from URL
                    # URL format: /cgi-bin/browse-edgar?action=getcompany&CIK=...
                    # or /Archives/edgar/data/CIK/ACCESSION/...
                    accession = self._extract_accession_from_url(href)

                    results.append(
                        {
                            "accession_number": accession,
                            "company_name": title.text if title.text else "",
                            "filed_date": (
                                updated.text[:10] if updated is not None else ""
                            ),
                            "form_type": "NPORT-P",
                            "url": href,
                        }
                    )
        except ET.ParseError as e:
            logger.error(f"Failed to parse Atom feed: {e}")

        return results

    def _extract_accession_from_url(self, url: str) -> str:
        """Extract accession number from SEC URL."""
        # Pattern: /Archives/edgar/data/1234567/000123456789012345/...
        parts = url.split("/")
        for i, part in enumerate(parts):
            if part == "data" and i + 2 < len(parts):
                return parts[i + 2]
        return ""

    def _search_by_cik(self, cik: str, form_type: str, count: int) -> list[dict]:
        """
        Search filings for a specific CIK using submissions API.

        Args:
            cik: The CIK number
            form_type: Form type to filter
            count: Maximum results

        Returns:
            List of filing metadata
        """
        cik_padded = cik.lstrip("0").zfill(10)
        submissions_url = (
            f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
        )

        try:
            response = self._make_request(submissions_url)
            data = response.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
            return []

        results = []
        filings = data.get("filings", {}).get("recent", {})

        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        primary_docs = filings.get("primaryDocument", [])

        company_name = data.get("name", "")

        for i, form in enumerate(forms):
            if form == form_type and len(results) < count:
                accession = accessions[i] if i < len(accessions) else ""
                results.append(
                    {
                        "accession_number": accession,
                        "cik": cik_padded,
                        "company_name": company_name,
                        "filed_date": dates[i] if i < len(dates) else "",
                        "form_type": form,
                        "primary_document": (
                            primary_docs[i] if i < len(primary_docs) else ""
                        ),
                    }
                )

        return results

    def download_filing(self, accession_number: str, cik: Optional[str] = None) -> str:
        """
        Download an N-PORT XML filing.

        Args:
            accession_number: The SEC accession number (e.g., "0001234567-89-012345")
            cik: Optional CIK number (required if not in cache)

        Returns:
            XML content of the filing

        Raises:
            ValueError: If filing cannot be found or downloaded
        """
        # Normalize accession number (remove dashes for URL)
        accession_clean = accession_number.replace("-", "")

        # Check cache first
        cache_key = hashlib.md5(accession_number.encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.xml"

        if cache_file.exists():
            logger.info(f"Loading filing from cache: {accession_number}")
            return cache_file.read_text(encoding="utf-8")

        # Need CIK to construct URL
        if cik is None:
            raise ValueError(
                "CIK required to download filing (not found in cache)"
            )

        cik_padded = cik.lstrip("0").zfill(10)

        # Try to find the primary XML document
        # N-PORT filings typically have primary document named like 'primary_doc.xml'
        index_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/"
            f"{accession_clean}/index.json"
        )

        try:
            response = self._make_request(index_url)
            index_data = response.json()

            # Find the primary XML document
            xml_doc = None
            for item in index_data.get("directory", {}).get("item", []):
                name = item.get("name", "")
                if name.endswith(".xml") and "primary" in name.lower():
                    xml_doc = name
                    break
                # Fallback: any XML that's not the index
                if name.endswith(".xml") and "index" not in name.lower():
                    xml_doc = name

            if xml_doc is None:
                # Try common N-PORT naming patterns
                xml_doc = "primary_doc.xml"

        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"Could not fetch filing index: {e}")
            xml_doc = "primary_doc.xml"

        # Download the XML
        xml_url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/"
            f"{accession_clean}/{xml_doc}"
        )

        try:
            response = self._make_request(xml_url)
            xml_content = response.text

            # Cache the downloaded filing
            cache_file.write_text(xml_content, encoding="utf-8")
            logger.info(f"Downloaded and cached filing: {accession_number}")

            return xml_content

        except requests.RequestException as e:
            raise ValueError(
                f"Failed to download filing {accession_number}: {e}"
            ) from e

    def parse_nport_xml(self, xml_content: str) -> Optional[Filing]:
        """
        Parse N-PORT XML content to extract fund and position data.

        Args:
            xml_content: Raw XML string from N-PORT filing

        Returns:
            Filing object with parsed data, or None if parsing fails
        """
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return None

        # Detect namespace
        ns = self._detect_namespace(root)

        # Extract fund info
        fund_name = self._find_text(root, ".//genInfo/regName", ns) or self._find_text(
            root, ".//edgarSubmission/headerData/filerInfo/seriesClassInfo/seriesName", ns
        ) or "Unknown Fund"

        reporting_date = self._find_text(root, ".//genInfo/repPdDate", ns) or ""

        cik = self._find_text(root, ".//genInfo/regCik", ns) or ""

        accession = self._find_text(root, ".//accessionNumber", ns) or ""

        # Total assets
        total_assets_str = self._find_text(root, ".//fundInfo/totAssets", ns) or "0"
        try:
            total_assets = float(total_assets_str)
        except ValueError:
            total_assets = 0.0

        # Parse positions
        positions = []
        for inv_or_sec in root.iter():
            if inv_or_sec.tag.endswith("invstOrSec"):
                position = self._parse_position(inv_or_sec, ns)
                if position:
                    positions.append(position)

        return Filing(
            fund_name=fund_name,
            reporting_date=reporting_date,
            accession_number=accession,
            cik=cik,
            total_assets=total_assets,
            positions=positions,
        )

    def _detect_namespace(self, root: ET.Element) -> dict[str, str]:
        """Detect the XML namespace used in the document."""
        tag = root.tag
        if "{" in tag:
            ns_uri = tag[1 : tag.index("}")]
            return {"ns": ns_uri}
        return {}

    def _find_text(
        self, element: ET.Element, path: str, ns: dict[str, str]
    ) -> Optional[str]:
        """
        Find text at path, trying with and without namespace.

        Args:
            element: Root element to search from
            path: XPath-like path
            ns: Namespace dictionary

        Returns:
            Text content or None
        """
        # Try without namespace first (simple path)
        parts = path.lstrip(".").lstrip("/").split("/")
        current = element

        for part in parts:
            if not part:
                continue
            found = None
            # Try direct child match
            for child in current:
                local_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if local_name == part:
                    found = child
                    break
            if found is None:
                return None
            current = found

        return current.text if current is not None else None

    def _parse_position(
        self, element: ET.Element, ns: dict[str, str]
    ) -> Optional[Position]:
        """Parse a single investment/security position from XML."""
        # Helper to get child text
        def get_child_text(tag_name: str) -> Optional[str]:
            for child in element.iter():
                local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if local == tag_name:
                    return child.text
            return None

        asset_name = get_child_text("name") or get_child_text("title") or ""
        cusip = get_child_text("cusip")
        lei = get_child_text("lei")
        asset_cat = get_child_text("assetCat") or ""

        # Get market value
        val_str = get_child_text("valUSD") or get_child_text("val") or "0"
        try:
            market_value = float(val_str)
        except ValueError:
            market_value = 0.0

        # Get percentage of portfolio
        pct_str = get_child_text("pctVal") or "0"
        try:
            pct_val = float(pct_str)
        except ValueError:
            pct_val = 0.0

        # Get units/quantity
        units_str = get_child_text("balance") or get_child_text("units")
        units = None
        if units_str:
            try:
                units = float(units_str)
            except ValueError:
                pass

        currency = get_child_text("curCd") or "USD"
        fair_value = get_child_text("fairValLevel")
        issuer = get_child_text("issuerName")

        return Position(
            asset_name=asset_name,
            cusip=cusip,
            lei=lei,
            asset_category=asset_cat,
            market_value=market_value,
            percentage_of_portfolio=pct_val,
            units=units,
            currency=currency,
            fair_value_level=fair_value,
            issuer_name=issuer,
        )

    def extract_clo_positions(self, filing_data: Filing) -> list[Position]:
        """
        Filter positions to only CLO and loan positions.

        Args:
            filing_data: Parsed Filing object

        Returns:
            List of Position objects that are CLO or loan related
        """
        clo_keywords = [
            "clo",
            "collateralized loan",
            "loan participation",
            "senior loan",
            "leveraged loan",
            "bank loan",
            "syndicated loan",
            "floating rate loan",
            "term loan",
        ]

        # Asset category codes that indicate loans
        loan_categories = [
            "LOAN",
            "DBT",  # Debt instruments
            "ABS-CBDO",  # Asset-backed securities - collateralized debt
        ]

        clo_positions = []

        for position in filing_data.positions:
            is_clo = False

            # Check asset category
            if position.asset_category.upper() in loan_categories:
                is_clo = True

            # Check name for CLO keywords
            name_lower = position.asset_name.lower()
            for keyword in clo_keywords:
                if keyword in name_lower:
                    is_clo = True
                    break

            # Check issuer name if available
            if position.issuer_name:
                issuer_lower = position.issuer_name.lower()
                for keyword in clo_keywords:
                    if keyword in issuer_lower:
                        is_clo = True
                        break

            if is_clo:
                clo_positions.append(position)

        logger.info(
            f"Found {len(clo_positions)} CLO/loan positions out of "
            f"{len(filing_data.positions)} total positions"
        )

        return clo_positions

    def calculate_ownership_metrics(
        self, positions: list[Position]
    ) -> OwnershipMetrics:
        """
        Calculate concentration metrics for a set of CLO positions.

        Args:
            positions: List of Position objects

        Returns:
            OwnershipMetrics with concentration calculations
        """
        if not positions:
            return OwnershipMetrics(
                total_clo_positions=0,
                total_market_value=0.0,
                average_position_size=0.0,
                top_5_concentration=0.0,
                herfindahl_index=0.0,
                largest_position_pct=0.0,
                positions_above_5pct=0,
            )

        total_value = sum(p.market_value for p in positions)
        n_positions = len(positions)

        # Sort by market value descending
        sorted_positions = sorted(
            positions, key=lambda p: p.market_value, reverse=True
        )

        # Top 5 concentration
        top_5_value = sum(p.market_value for p in sorted_positions[:5])
        top_5_pct = (top_5_value / total_value * 100) if total_value > 0 else 0

        # Herfindahl-Hirschman Index (sum of squared market shares)
        # Normalized to 0-1 range
        hhi = 0.0
        if total_value > 0:
            for p in positions:
                share = p.market_value / total_value
                hhi += share * share

        # Largest position percentage
        largest_pct = (
            (sorted_positions[0].market_value / total_value * 100)
            if total_value > 0
            else 0
        )

        # Count positions above 5% threshold
        above_5pct = sum(
            1
            for p in positions
            if total_value > 0 and (p.market_value / total_value * 100) >= 5
        )

        return OwnershipMetrics(
            total_clo_positions=n_positions,
            total_market_value=total_value,
            average_position_size=total_value / n_positions if n_positions > 0 else 0,
            top_5_concentration=top_5_pct,
            herfindahl_index=hhi,
            largest_position_pct=largest_pct,
            positions_above_5pct=above_5pct,
        )

    def get_clo_holdings_for_fund(
        self, cik: str, filing_count: int = 1
    ) -> list[dict[str, Any]]:
        """
        Convenience method to get CLO holdings for a fund.

        Args:
            cik: The fund's CIK number
            filing_count: Number of recent filings to process

        Returns:
            List of dicts with filing info and CLO positions
        """
        results = []

        # Search for filings
        filings = self.search_nport_filings(cik=cik, count=filing_count)

        for filing_meta in filings:
            accession = filing_meta.get("accession_number")
            if not accession:
                continue

            try:
                # Download and parse
                xml_content = self.download_filing(accession, cik=cik)
                filing = self.parse_nport_xml(xml_content)

                if filing is None:
                    continue

                # Extract CLO positions
                clo_positions = self.extract_clo_positions(filing)
                metrics = self.calculate_ownership_metrics(clo_positions)

                results.append(
                    {
                        "fund_name": filing.fund_name,
                        "reporting_date": filing.reporting_date,
                        "accession_number": accession,
                        "total_positions": len(filing.positions),
                        "clo_positions": len(clo_positions),
                        "clo_market_value": metrics.total_market_value,
                        "top_5_concentration": metrics.top_5_concentration,
                        "herfindahl_index": metrics.herfindahl_index,
                        "positions": clo_positions,
                        "metrics": metrics,
                    }
                )

            except (ValueError, requests.RequestException) as e:
                logger.error(f"Error processing filing {accession}: {e}")
                continue

        return results


if __name__ == "__main__":
    # Demonstration of EDGAR Parser usage
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 60)
    print("SEC EDGAR N-PORT Parser - CLO Holdings Extractor")
    print("=" * 60)

    # Initialize parser with a proper User-Agent
    # Note: Replace with your actual contact email for production use
    parser = EDGARParser(
        user_agent="LoanLiquidityPredictor/1.0 research@example.com"
    )

    # Example: BlackRock's CIK (one of the largest CLO holders)
    # CIK 0001006249 is iShares Trust
    example_cik = "1006249"

    print(f"\nSearching for N-PORT filings for CIK: {example_cik}")
    print("-" * 40)

    try:
        # Search for recent filings
        filings = parser.search_nport_filings(
            cik=example_cik,
            form_type="NPORT-P",
            count=3,
        )

        if not filings:
            print("No filings found. This could be due to:")
            print("  - Invalid CIK number")
            print("  - SEC API rate limiting")
            print("  - Network issues")
            sys.exit(0)

        print(f"Found {len(filings)} filings:")
        for f in filings:
            print(f"  - {f.get('filed_date', 'N/A')}: {f.get('accession_number', 'N/A')}")
            print(f"    Company: {f.get('company_name', 'N/A')}")

        # Process the most recent filing
        if filings:
            latest = filings[0]
            accession = latest.get("accession_number")

            if accession:
                print(f"\nDownloading filing: {accession}")
                print("-" * 40)

                xml_content = parser.download_filing(accession, cik=example_cik)
                print(f"Downloaded {len(xml_content):,} bytes")

                # Parse the XML
                filing = parser.parse_nport_xml(xml_content)

                if filing:
                    print(f"\nFund: {filing.fund_name}")
                    print(f"Reporting Date: {filing.reporting_date}")
                    print(f"Total Assets: ${filing.total_assets:,.2f}")
                    print(f"Total Positions: {len(filing.positions)}")

                    # Extract CLO positions
                    clo_positions = parser.extract_clo_positions(filing)
                    print(f"\nCLO/Loan Positions Found: {len(clo_positions)}")

                    if clo_positions:
                        # Show top 5 CLO positions
                        print("\nTop 5 CLO/Loan Positions:")
                        print("-" * 40)
                        sorted_clo = sorted(
                            clo_positions,
                            key=lambda p: p.market_value,
                            reverse=True,
                        )
                        for i, pos in enumerate(sorted_clo[:5], 1):
                            print(f"{i}. {pos.asset_name[:50]}")
                            print(f"   CUSIP: {pos.cusip or 'N/A'}")
                            print(f"   Value: ${pos.market_value:,.2f}")
                            print(f"   % of Portfolio: {pos.percentage_of_portfolio:.2f}%")

                        # Calculate metrics
                        metrics = parser.calculate_ownership_metrics(clo_positions)
                        print("\nOwnership Concentration Metrics:")
                        print("-" * 40)
                        print(f"Total CLO Positions: {metrics.total_clo_positions}")
                        print(f"Total CLO Value: ${metrics.total_market_value:,.2f}")
                        print(f"Average Position: ${metrics.average_position_size:,.2f}")
                        print(f"Top 5 Concentration: {metrics.top_5_concentration:.1f}%")
                        print(f"Herfindahl Index: {metrics.herfindahl_index:.4f}")
                        print(f"Largest Position: {metrics.largest_position_pct:.1f}%")
                        print(f"Positions >= 5%: {metrics.positions_above_5pct}")
                    else:
                        print("No CLO/loan positions found in this filing.")
                else:
                    print("Failed to parse filing XML.")

    except requests.RequestException as e:
        print(f"Network error: {e}")
        print("Please check your internet connection and try again.")
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Demo complete.")
