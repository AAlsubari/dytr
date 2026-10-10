import urllib.request
import urllib.parse
import re
import json
import gzip
import ssl
from bs4 import BeautifulSoup

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept-Encoding": "gzip, deflate",
}

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

def fetch_html_stealth(url):
    req = urllib.request.Request(url, headers=_BROWSER_HEADERS)
    with urllib.request.urlopen(req, timeout=12, context=_SSL_CTX) as resp:
        raw = resp.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors="ignore")

def extract_single_page_records_universal(html_content, extract_mode="auto", base_url=""):
    if not html_content or not html_content.strip():
        return [], [], "empty"

    soup = BeautifulSoup(html_content, "html.parser")
    # Decompose script, style, noscript, etc.
    for junk in soup(["script", "style", "noscript", "meta", "svg", "form", "button", "nav", "footer", "iframe"]):
        junk.decompose()

    # 1. SPECIALIZED EXTRACTOR: arXiv Scientific Papers (<dl><dt><dd>)
    dls = soup.find_all("dl")
    if dls and any(dl.find(["dt", "dd"]) for dl in dls):
        paper_rows = []
        for dl in dls:
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                id_tag = dt.find("span", class_=re.compile(r"list-identifier"))
                id_link = id_tag.find("a") if id_tag else dt.find("a")
                paper_id = id_tag.get_text(strip=True) if id_tag else (dt.get_text(strip=True)[:25])
                link = urllib.parse.urljoin(base_url, id_link["href"]) if id_link and id_link.get("href") else ""
                
                title_tag = dd.find(["div", "span"], class_=re.compile(r"list-title|title"))
                title = title_tag.get_text(strip=True).replace("Title:", "").strip() if title_tag else ""
                if not title:
                    first_line = dd.get_text(strip=True).split("\n")[0]
                    title = first_line[:150]
                
                authors_tag = dd.find(["div", "span"], class_=re.compile(r"list-authors|authors"))
                authors = authors_tag.get_text(strip=True).replace("Authors:", "").strip() if authors_tag else ""
                
                subjects_tag = dd.find(["div", "span"], class_=re.compile(r"list-subjects|subjects"))
                subjects = subjects_tag.get_text(strip=True).replace("Subjects:", "").strip() if subjects_tag else ""

                if title or paper_id:
                    paper_rows.append({
                        "item_id": len(paper_rows) + 1,
                        "paper_id": paper_id,
                        "title": title,
                        "authors": authors,
                        "subjects": subjects,
                        "url": link
                    })
        if len(paper_rows) >= 2:
            return ["item_id", "paper_id", "title", "authors", "subjects", "url"], paper_rows, "arxiv_papers"

    # 2. SPECIALIZED EXTRACTOR: Telegram Web Channel Posts (t.me/s/...)
    tg_wraps = soup.find_all("div", class_=re.compile(r"tgme_widget_message_wrap"))
    if len(tg_wraps) >= 2:
        tg_rows = []
        for idx, wrap in enumerate(tg_wraps):
            txt_el = wrap.find("div", class_=re.compile(r"tgme_widget_message_text"))
            date_el = wrap.find("time") or wrap.find("a", class_=re.compile(r"tgme_widget_message_date"))
            views_el = wrap.find("span", class_=re.compile(r"tgme_widget_message_views"))
            link_el = wrap.find("a", class_=re.compile(r"tgme_widget_message_date"))
            
            text = txt_el.get_text(strip=True) if txt_el else ""
            date_str = date_el.get_text(strip=True) if date_el else ""
            views = views_el.get_text(strip=True) if views_el else ""
            link = link_el["href"] if link_el and link_el.get("href") else ""

            if text or date_str:
                tg_rows.append({
                    "post_id": idx + 1,
                    "date": date_str,
                    "views": views,
                    "text": text,
                    "link": link
                })
        if tg_rows:
            return ["post_id", "date", "views", "text", "link"], tg_rows, "telegram_posts"

    # 3. HTML TABLES EXTRACTOR (Standard <table> elements)
    if extract_mode in ["tables", "auto"]:
        tables = soup.find_all("table")
        for table in tables:
            trs = table.find_all("tr")
            if len(trs) >= 2:
                # Header resolution
                header_cells = trs[0].find_all(["th", "td"])
                raw_cols = [c.get_text(strip=True) for c in header_cells]
                cols = [c if c else f"Col_{i+1}" for i, c in enumerate(raw_cols)]
                
                table_rows = []
                for r in trs[1:]:
                    cells = r.find_all(["td", "th"])
                    vals = [c.get_text(strip=True) for c in cells]
                    if vals and any(len(v) > 0 for v in vals):
                        row_dict = {}
                        for i, val in enumerate(vals):
                            c_name = cols[i] if i < len(cols) else f"Col_{i+1}"
                            row_dict[c_name] = val
                        table_rows.append(row_dict)
                
                if len(table_rows) >= 2:
                    return cols, table_rows, "tables"

    # 4. REPEATING CARDS / ITEMS / LIST ITEMS (General articles, products, feeds)
    if extract_mode in ["cards", "auto"]:
        card_elements = soup.find_all(["article", "section", "li", "div"], class_=re.compile(r"(?i)(card|item|product|listing|row|entry|post|result|box|search|book|article)"))
        
        # Also check child elements of main containers if class match is weak
        if len(card_elements) < 3:
            for container in soup.find_all(["ul", "ol", "main", "div"], id=re.compile(r"(?i)(results|items|list|content|search)")):
                children = [c for c in container.find_all(["li", "div", "article"], recursive=False)]
                if len(children) >= 3:
                    card_elements.extend(children)

        # Deduplicate & parse cards
        parsed_cards = []
        for idx, el in enumerate(card_elements):
            title_el = el.find(["h1", "h2", "h3", "h4", "h5", "strong", "b"]) or el.find("a")
            link_el = el.find("a", href=True)
            desc_el = el.find(["p", "div", "span"], class_=re.compile(r"(?i)(desc|summary|text|snippet|detail|abstract)"))
            meta_el = el.find(class_=re.compile(r"(?i)(meta|badge|tag|price|date|author|score|comment|byline)"))
            
            title = title_el.get_text(strip=True) if title_el else ""
            link = urllib.parse.urljoin(base_url, link_el["href"].strip()) if link_el else ""
            desc = desc_el.get_text(strip=True) if desc_el else ""
            meta = meta_el.get_text(strip=True) if meta_el else ""

            # If no desc tag, grab first long text
            if not desc:
                for p in el.find_all("p"):
                    p_txt = p.get_text(strip=True)
                    if len(p_txt) > 20 and p_txt != title:
                        desc = p_txt
                        break

            if len(title) >= 3 or (len(desc) >= 20 and link):
                parsed_cards.append({
                    "item_id": len(parsed_cards) + 1,
                    "title": title[:200],
                    "meta_info": meta[:150],
                    "description": desc[:300],
                    "url": link
                })

        if len(parsed_cards) >= 2:
            return ["item_id", "title", "meta_info", "description", "url"], parsed_cards[:100], "cards"

    # 5. UNIVERSAL PARAGRAPHS & TEXT EXTRACTION FALLBACK
    text_blocks = []
    for p_idx, p in enumerate(soup.find_all(["p", "article", "blockquote", "div"])):
        txt = p.get_text(strip=True)
        if len(txt) >= 25 and len(txt.split()) >= 4:
            # Check if this text is not already covered by parent
            text_blocks.append({
                "record_id": len(text_blocks) + 1,
                "text_content": txt,
                "word_count": len(txt.split()),
                "char_count": len(txt)
            })

    if text_blocks:
        return ["record_id", "text_content", "word_count", "char_count"], text_blocks[:100], "text_blocks"

    return [], [], "none"

def test_all():
    urls = [
        ("arXiv AI", "https://arxiv.org/list/cs.AI/recent?skip=0&show=25"),
        ("Telegram Channel", "https://t.me/s/Jamal_Almansoub_Channel"),
        ("Wikipedia GDP", "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"),
    ]

    for name, u in urls:
        print(f"\n==================== {name} ====================")
        try:
            html = fetch_html_stealth(u)
            cols, rows, mode = extract_single_page_records_universal(html, "auto", u)
            print(f"SUCCESS: Mode='{mode}' | Cols={cols} | Total Rows={len(rows)}")
            if rows:
                print(f"Row 1 Title: {str(rows[0].get('title') or rows[0].get('paper_id') or rows[0].get('text') or rows[0].get('text_content') or rows[0])[:80]}")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    test_all()
