import bs4
import os
import glob
import json

folder_path = os.path.dirname(os.path.abspath(__file__))
html_files = glob.glob(os.path.join(folder_path, "*.html"))

parsed_files = []
law_content = []
for file_path in html_files:

    with open(file_path, 'r', encoding='utf-8') as f:
        soup = bs4.BeautifulSoup(f.read(), 'html.parser')
        content = soup.find('div', class_='law-reg-content')
        law_name = soup.find('a', id='hlLawName').get_text(strip=True)
        if content:
            rows = content.find_all('div',class_='row')
            for row in rows:
                col_no = row.find('div', class_='col-no')
                col_data = row.find('div', class_='col-data')
                if col_data and col_no:
                    article_no = col_no.get_text(strip=True)
                    law_text = col_data.get_text(strip=True)
                    law_content.append((law_name, article_no, law_text))

# After the loop, save to JSON
with open('law_data.json', 'w', encoding='utf-8') as f:
    json_data = [{"law_name": l[0], "article_no": l[1], "law_text": l[2]} for l in law_content]
    json.dump(json_data, f, ensure_ascii=False, indent=2)
