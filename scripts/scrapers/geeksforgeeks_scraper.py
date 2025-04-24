import requests
from bs4 import BeautifulSoup
import json


def get_main_algorithm_categories(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Navigate to the div with id 'header-main_slider-outer-div'
    outer_div = soup.find('div', id='header-main__slider-outer-div')
    if not outer_div:
        print("Could not find the outer div.")
        return []

    # Then go to the inner slider
    slider_div = outer_div.find('div', class_='header-main__slider')
    if not slider_div:
        print("Could not find the slider div.")
        return []

    # Then access the ul with id 'hslider'
    ul_tag = slider_div.find('ul', id='hslider')
    if not ul_tag:
        print("Could not find the <ul> with id 'hslider'.")
        return []

    # Now get all the category <a> links
    links = ul_tag.find_all('a')
    categories = [(link.text.strip(), link['href']) for link in links if link.get('href')]

    return categories



def get_algorithm_links(category_url):
    """Extract algorithm links from a category page."""
    response = requests.get(category_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    ul_tag = soup.find('ul', id='hslider')
    if not ul_tag:
        print(f"Could not find the <ul> with id 'hslider' on {category_url}.")
        return []

    links = ul_tag.find_all('a')
    algorithm_links = [link['href'] for link in links if 'href' in link.attrs]

    return algorithm_links


def scrape_algorithm_page(url, category_name):
    """Scrape detailed information from an individual algorithm page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    title = soup.find('h1').text.strip() if soup.find('h1') else 'No Title'
    
    description = ''
    for p in soup.find_all('p'):
        description += p.text.strip() + ' '
    
    code = ''
    code_blocks = soup.find_all('pre')
    for block in code_blocks:
        code += block.text.strip() + '\n'
    
    complexity = {'time': '', 'space': ''}
    for p in soup.find_all('p'):
        if 'time complexity' in p.text.lower():
            complexity['time'] = p.text.strip()
        elif 'space complexity' in p.text.lower():
            complexity['space'] = p.text.strip()
    
    tags = []
    tag_section = soup.find('div', class_='entry-meta')
    if tag_section:
        tag_links = tag_section.find_all('a')
        tags = [tag.text.strip() for tag in tag_links]
    else:
        tags = [category_name]  # Use the last part of the URL as a tag
    
    return {
        'title': title,
        'description': description,
        'code': code,
        'complexity': complexity,
        'tags': tags,
        'url': url
    }

def scrape_all_algorithms(base_url):
    """Scrape all algorithms from the base URL."""
    categories = get_main_algorithm_categories(base_url)
    all_algorithms = []

    for category_name, category_url in categories:
        print(f"Scraping category: {category_name}")
        algorithm_links = get_algorithm_links(category_url)
        for link in algorithm_links:
            print(f"Scraping algorithm: {link}")
            algorithm_data = scrape_algorithm_page(link, category_name)
            all_algorithms.append(algorithm_data)

    return all_algorithms

def save_to_json(data, filename='./data/raw/geeksforgeeks_algorithms.json'):
    """Save the scraped data to a JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


base_url = 'https://www.geeksforgeeks.org/fundamentals-of-algorithms/'
algorithms = scrape_all_algorithms(base_url)
save_to_json(algorithms)