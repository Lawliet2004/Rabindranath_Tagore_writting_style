"""Conservative public-page collection. Network calls occur only on explicit invocation."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import logging
import re
import time
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

from .common import atomic_bytes, canonical_hash, file_hash, now, read_json, read_jsonl, sha256, write_json, write_jsonl

LOG = logging.getLogger(__name__)
ORIGIN = "https://bn.wikisource.org"
AUTHOR = "রবীন্দ্রনাথ ঠাকুর"
GENRES = {"কাব্য": "poems", "কবিতা": "poems", "কাব্যগ্রন্থ": "poems", "গান": "songs",
          "গীতিনাট্য": "plays", "ছোটগল্প": "short_stories", "গল্প": "short_stories",
          "উপন্যাস": "novels", "নাটক": "plays", "প্রবন্ধ": "essays",
          "সমালোচনা": "criticism", "পত্র": "letters_prose", "চিঠিপত্র": "letters_prose"}


class PolicyError(RuntimeError):
    pass


def encoded_url(url: str) -> str:
    s = urlsplit(url)
    return urlunsplit((s.scheme, s.netloc, quote(unquote(s.path), safe="/():"), s.query, ""))


class PoliteClient:
    """One serial connection; robots checked, 429/503 respected, immutable local cache."""
    def __init__(self, root: Path, contact: str, delay=3.0):
        import requests
        if not contact or not ("@" in contact or contact.startswith("https://")):
            raise ValueError("Set collection.contact to your real contact email or project URL before collection.")
        self.root = root
        self.cache = root / "data/raw/http"
        self.session = requests.Session()
        self.agent = f"TagoreCorpusBot/0.1 ({contact})"
        self.session.headers.update({"User-Agent": self.agent})
        self.delay = max(3.0, float(delay))
        self.last = 0.0
        self.robots = None
        self._load_robots()

    def _load_robots(self):
        url = ORIGIN + "/robots.txt"
        response = self.session.get(url, timeout=(10, 45), allow_redirects=False)
        if response.status_code != 200:
            raise PolicyError(f"Could not verify robots.txt (HTTP {response.status_code}); collection stopped.")
        body = response.content
        atomic_bytes(self.root / "data/metadata/robots-wikisource.txt", body)
        write_json(self.root / "data/metadata/robots-audit.json", {
            "url": url, "fetched_at": now(), "sha256": sha256(body), "user_agent": self.agent})
        self.robots = RobotFileParser()
        self.robots.parse(body.decode("utf-8-sig").splitlines())
        self.delay = max(self.delay, self.robots.crawl_delay(self.agent) or self.robots.crawl_delay("*") or 0)
        self.last = time.monotonic()

    def get(self, url: str) -> tuple[bytes, dict]:
        url = encoded_url(url)
        if urlsplit(url).netloc != "bn.wikisource.org":
            raise PolicyError(f"Unapproved host: {url}")
        if not self.robots.can_fetch(self.agent, url):
            raise PolicyError(f"robots.txt disallows {url}; use an authorized manual import.")
        key = sha256(url)
        path = self.cache / f"{key}.html"
        meta = self.cache / f"{key}.json"
        if path.is_file() and meta.is_file():
            record = read_json(meta)
            if file_hash(path) != record["sha256"]:
                raise IOError(f"Corrupted cached download: {path}. Restore or remove this cache entry.")
            return path.read_bytes(), record
        for attempt in range(5):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last)))
            response = self.session.get(url, timeout=(10, 60), allow_redirects=False)
            self.last = time.monotonic()
            if response.status_code in (429, 503):
                retry = response.headers.get("Retry-After", "")
                if retry.isdigit():
                    delay = max(int(retry), 5 * 2**attempt)
                elif retry:
                    from email.utils import parsedate_to_datetime
                    from datetime import datetime, timezone
                    delay = max(5 * 2**attempt, (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds())
                else:
                    delay = 5 * 2**attempt
                LOG.warning("HTTP %s; backing off %.0fs", response.status_code, delay)
                time.sleep(delay)
                continue
            # Redirects are not silently followed to an unchecked host or path.
            if response.is_redirect:
                raise PolicyError(f"Redirect at {url}; review Location={response.headers.get('Location')!r} before adding it.")
            if response.status_code in (401, 403):
                raise PolicyError(f"Access denied (HTTP {response.status_code}) at {url}; collection stopped without bypassing restrictions.")
            response.raise_for_status()
            if "html" not in response.headers.get("Content-Type", ""):
                raise ValueError(f"Expected HTML at {url}; received {response.headers.get('Content-Type')}")
            if len(response.content) > 15 * 1024**2:
                raise ValueError(f"Page exceeds 15 MiB safety limit: {url}; import in reviewed sections.")
            record = {"url": url, "fetched_at": now(), "sha256": sha256(response.content),
                      "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                      "raw_path": str(path.relative_to(self.root)), "user_agent": self.agent}
            atomic_bytes(path, response.content)
            write_json(meta, record)
            return response.content, record
        raise RuntimeError(f"Repeated throttling at {url}; stop and retry on a later day.")


def wiki_title(href: str) -> str | None:
    s = urlsplit(urljoin(ORIGIN, href))
    if s.netloc != "bn.wikisource.org" or not s.path.startswith("/wiki/") or s.query:
        return None
    title = unquote(s.path[len("/wiki/"):]).replace("_", " ")
    # Mainspace only: namespaces such as রচনা:, নির্ঘণ্ট:, পাতা: are discovery aids, not training text.
    return title if ":" not in title else None


def canonical_work(title: str) -> str:
    title = title.split("/")[0]
    title = re.sub(r"\s*\([০-৯0-9]{4}[^)]*\)", "", title)
    return "bn-work-" + sha256(re.sub(r"\s+", " ", title).strip())[:20]


def discover_catalog(html: bytes):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".mw-parser-output")
    if body is None:
        raise ValueError("Wikisource catalog DOM changed: .mw-parser-output is absent.")
    for n in body.select(".navbox, .sistersitebox, .mw-editsection, .authority-control, #toc"):
        n.decompose()
    genre = "unknown"
    seen = set()
    for tag in body.find_all(["h2", "h3", "h4", "a"]):
        if tag.name.startswith("h"):
            heading = tag.get_text(" ", strip=True)
            genre = next((g for word, g in GENRES.items() if word in heading), "unknown")
        elif "new" not in tag.get("class", []):
            title = wiki_title(tag.get("href", ""))
            if title and title not in seen:
                seen.add(title)
                yield {"title": title, "genre": genre, "work_id": canonical_work(title)}


def extract_page(html: bytes, title: str) -> tuple[str, dict, list[str]]:
    """Preserve text nodes, explicit BRs and paragraph/stanza boundaries; remove known UI only."""
    from bs4 import BeautifulSoup, NavigableString
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".mw-parser-output")
    if body is None:
        raise ValueError("No literary content container; page requires manual review.")
    page_text = body.get_text(" ", strip=True)
    headers = body.select(".header, .ws-header, .headertemplate, .ws-author")
    author_verified = any(AUTHOR == re.sub(r"\s+", " ", a.get_text(" ", strip=True))
                          for header in headers for a in header.select("a"))
    categories = soup.select_one(".catlinks")
    author_verified = author_verified or bool(categories and "রবীন্দ্রনাথ ঠাকুর রচিত" in categories.get_text(" ", strip=True))
    rights_links = [a.get("href", "") for a in soup.select("a[href]")
                    if "creativecommons.org/licenses/by-sa/" in a.get("href", "")]
    pd_evidence = any(x in page_text for x in ["পাবলিক ডোমেইন", "পাবলিক ডোমেইনে", "কপিরাইটের মেয়াদ উত্তীর্ণ", "public domain"])
    children = sorted({t for a in body.select("a[href]") if (t := wiki_title(a["href"])) and t.startswith(title + "/")})
    roots = body.select(".prp-pages-output")
    output = BeautifulSoup("<div></div>", "html.parser").div
    if roots:
        for r in roots:
            output.append(BeautifulSoup(str(r), "html.parser"))
    else:
        output = body
    removed = {}
    for selector in ["script", "style", "#toc", ".toc", ".header", ".ws-header", ".ws-noexport", ".noprint", ".noexport",
                     ".navbox", ".navigation", ".catlinks", ".mw-editsection", ".pagenum", ".ws-pagenum", ".prp-pagequalityheader",
                     ".licenseContainer", ".licensetpl", ".license", ".ws-license", ".printfooter", "sup.reference", ".references", ".reflist"]:
        nodes = output.select(selector)
        removed[selector] = len(nodes)
        for n in nodes:
            n.decompose()
    warnings = []
    if output.select("table"):
        warnings.append("table_in_literary_container")
    # Unexpected front matter or editorial material is quarantined, not rewritten.
    if any(w in output.get_text(" ", strip=True)[:1500] for w in ["সম্পাদকের", "প্রকাশকের নিবেদন", "সূচিপত্র", "গ্রন্থপরিচয়"]):
        warnings.append("possible_editorial_or_front_matter")

    def render(node):
        if isinstance(node, NavigableString):
            return str(node)
        if node.name == "br":
            return "\n"
        inside = "".join(render(c) for c in node.children)
        if node.name in ("p", "h1", "h2", "h3", "h4", "blockquote", "section"):
            return "\n\n" + inside.strip("\n") + "\n\n"
        if node.name == "div" and "poem" in node.get("class", []):
            return "\n\n" + inside.strip("\n") + "\n\n"
        return inside

    text = render(output)
    meta = {"author_verified": author_verified, "rights_links": rights_links, "public_domain_notice": pd_evidence,
            "extraction": {"container": "prp-pages-output" if roots else "mw-parser-output", "removed_elements": removed},
            "extraction_warnings": warnings, "is_container": bool(children)}
    return text, meta, children


def collect(root: Path, config: dict):
    from tqdm.auto import tqdm
    settings = config["collection"]
    client = PoliteClient(root, settings["contact"], settings["delay_seconds"])
    catalog_url = ORIGIN + "/wiki/" + quote(settings["author_title"].replace(" ", "_"))
    html, catalog_record = client.get(catalog_url)
    seeds = list(discover_catalog(html))
    write_jsonl(root / "data/metadata/source_discovery.jsonl", seeds)
    queue = deque((s, 0) for s in seeds)
    seen = set()
    events = []
    bar = tqdm(total=settings["max_pages"], desc="Cached / collected pages")
    while queue and len(seen) < settings["max_pages"]:
        item, depth = queue.popleft()
        title = item["title"]
        if title in seen:
            continue
        seen.add(title)
        url = ORIGIN + "/wiki/" + quote(title.replace(" ", "_"), safe="/")
        try:
            html, record = client.get(url)
            text, extraction, children = extract_page(html, title)
            doc_id = "ws-" + sha256(url)[:24]
            doc = {"doc_id": doc_id, "title": title, "genre": item["genre"], "work_id": item["work_id"],
                   "source_url": url, "author": AUTHOR, "language": "bn", "edition": None, "publication_date": None,
                   "rights_status": "public-domain-notice; Wikisource transcription CC-BY-SA" if extraction["public_domain_notice"] else "Wikisource transcription license; underlying edition requires review",
                   "rights_evidence_url": extraction["rights_links"][0] if extraction["rights_links"] else url,
                   "rights_verified": bool(extraction["public_domain_notice"] and extraction["rights_links"]),
                   "catalog_url": catalog_url, "catalog_checksum": catalog_record["sha256"],
                   "content_checksum": sha256(text), "download": record, "text": text, **extraction}
            # Containers are retained raw for audit, but excluded from training to avoid TOC/frontmatter/child duplication.
            write_json(root / "data/raw/documents" / f"{doc_id}.json", doc)
            events.append({"title": title, "status": "cached_or_downloaded", "doc_id": doc_id})
            if depth < settings["max_depth"]:
                for child in children:
                    queue.append(({**item, "title": child}, depth + 1))
        except PolicyError:
            raise
        except Exception as e:
            LOG.exception("Could not process %s", title)
            events.append({"title": title, "status": "error", "error": str(e)})
        finally:
            bar.update(1)
            write_jsonl(root / "data/metadata/collection_events.jsonl", events)
    bar.close()
    return {"catalog_candidates": len(seeds), "visited": len(seen), "events": len(events)}


def import_manual(root: Path, manifest: Path):
    """Only explicitly authorized UTF-8 originals; retain exact source bytes and checksums."""
    result = []
    for row in read_jsonl(manifest):
        required = ["path", "title", "genre", "author", "language", "source_url", "work_id", "rights_status", "rights_evidence_url"]
        if any(not row.get(k) or "REPLACE_" in str(row[k]) for k in required):
            raise ValueError(f"Incomplete manual-import metadata: {row.get('title')}")
        if row.get("authorized") is not True or row.get("reviewed_original_text") is not True:
            raise ValueError("Manual imports require authorized=true and reviewed_original_text=true.")
        if row["author"] != AUTHOR or row["language"] != "bn":
            raise ValueError("Only Tagore's original Bengali texts belong in this corpus.")
        path = Path(row["path"]).expanduser().resolve()
        raw = path.read_bytes()
        text = raw.decode("utf-8")  # No errors=replace: broken UTF-8 must be repaired from the source.
        doc_id = "manual-" + canonical_hash({"source": row["source_url"], "checksum": sha256(raw)})[:24]
        target = root / "data/raw/manual" / f"{doc_id}.txt"
        atomic_bytes(target, raw)
        doc = {**row, "doc_id": doc_id, "text": text, "author_verified": True, "rights_verified": True,
               "edition": row.get("edition"), "publication_date": row.get("publication_date"),
               "is_container": False, "extraction_warnings": [], "content_checksum": sha256(text),
               "download": {"raw_path": str(target.relative_to(root)), "sha256": sha256(raw), "imported_at": now()}}
        write_json(root / "data/raw/documents" / f"{doc_id}.json", doc)
        result.append(doc_id)
    return result
