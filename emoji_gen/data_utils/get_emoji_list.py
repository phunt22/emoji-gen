import requests
from bs4 import BeautifulSoup
import re
import json
from pathlib import Path

def main():
    url = 'https://emojigraph.org/apple/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    all_emoji_data = []

    for div_num in range(7, 16):
        selector = f'#category__first > div > div > div.col-12.col-lg-8 > div:nth-child({div_num})'
        category_div = soup.select_one(selector)

        if category_div:
            for img in category_div.find_all('img'):
                if 'src' in img.attrs:
                    path = img['src']
                    name_match = re.search(r'/([^/]+)_[^/]+\.png$', path)
                    if name_match:
                        name = name_match.group(1)
                        processed_url = path.replace('/72/', '/')
                        full_url = f"https://emojigraph.org{processed_url}"
                        processed_name = name.replace('-', ' ') + ' emoji'

                        all_emoji_data.append({
                            'link': full_url,
                            'name': name,
                            'processed': processed_name,
                        })

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / 'emojis.json', 'w', encoding='utf-8') as f:
        json.dump(all_emoji_data, f, ensure_ascii=False, indent=2)
