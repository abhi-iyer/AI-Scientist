import os
import json
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

'''
configuration stuff
'''
SAVE_DIR = "biorxiv_papers"
TEXT_DIR = os.path.join(SAVE_DIR, "full_texts")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

CATEGORIES = ["neuroscience"]
YEARS = list(range(2014, 2025))  # Get papers from 2014 to present
MAX_PAPERS_PER_YEAR = 10000  # limit to 10000 papers per year

API_URL_TEMPLATE = "https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}/json"


'''
fetching papers from biorxiv
'''
def fetch_biorxiv_papers():
    papers = []

    for year in YEARS:
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        cursor = 0  # paginate with biorxiv's cursor

        while True:
            api_url = API_URL_TEMPLATE.format(start_date=start_date, end_date=end_date, cursor=cursor)

            response = requests.get(api_url)
            if response.status_code != 200:
                print(f"Error fetching BioRxiv data for {year} (Cursor {cursor}): {response.status_code}")
                break  # stop if error

            data = response.json()
            results = data.get("collection", [])

            # breaking condition
            if not results or cursor > MAX_PAPERS_PER_YEAR:
                break

            for paper in tqdm(results, desc=f"Fetching BioRxiv {year} (Cursor {cursor})"):
                if paper.get("category", "").lower() in CATEGORIES:  # only pick topics in CATEGORIES
                    try:
                        jatsxml_url = paper.get("jatsxml", "")

                        papers.append({
                            "doi": paper["doi"],
                            "title": paper["title"],
                            "authors": paper["authors"],
                            "date": paper["date"],
                            "version": paper["version"],
                            "category": paper["category"],
                            "abstract": paper["abstract"],
                            "jatsxml_url": jatsxml_url,
                            "url": f"https://www.biorxiv.org/content/{paper['doi']}.v{paper['version']}"
                        })
                    except Exception as e:
                        print(f"Error processing paper {paper['doi']}: {e}")

            cursor += len(results)  # move to next page

    return papers


'''
extract text from JATS XML
'''
def extract_text_from_jatsxml(jatsxml_url):
    if not jatsxml_url:
        return ""

    try:
        response = requests.get(jatsxml_url)
        if response.status_code != 200:
            print(f"Failed to fetch JATS XML: {jatsxml_url}")
            return ""

        soup = BeautifulSoup(response.content, "lxml-xml")

        # extract main body text
        sections = soup.find_all("sec")
        full_text = "\n\n".join([sec.get_text(separator=" ", strip=True) for sec in sections])

        return full_text
    except Exception as e:
        print(f"Error extracting JATS XML text: {e}")
        return ""


'''
runner
'''
def main():
    print(f"Fetching papers from {YEARS[0]} to {YEARS[-1]}...")
    papers = fetch_biorxiv_papers()

    print(f"Extracting text from JATS XML...")
    for paper in tqdm(papers, desc="Processing Papers"):
        full_text = extract_text_from_jatsxml(paper["jatsxml_url"])

        if full_text:
            # save extracted text separately
            text_filename = paper["doi"].replace("/", "_") + ".txt"
            text_path = os.path.join(TEXT_DIR, text_filename)
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            paper["full_text"] = full_text

    # save metadata with full text
    json_path = os.path.join(SAVE_DIR, "biorxiv_neuroscience_jats.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=4)

    print(f"Saved {len(papers)} neuroscience papers with full text to {json_path}")


if __name__ == "__main__":
    main()