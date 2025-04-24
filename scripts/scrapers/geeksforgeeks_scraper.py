import requests
from bs4 import BeautifulSoup
import json


def get_main_algorithm_categories(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    outer_div = soup.find('div', id='header-main__slider-outer-div')
    if not outer_div:
        print("Could not find the outer div.")
        return []

    slider_div = outer_div.find('div', class_='header-main__slider')
    if not slider_div:
        print("Could not find the slider div.")
        return []

    ul_tag = slider_div.find('ul', id='hslider')
    if not ul_tag:
        print("Could not find the <ul> with id 'hslider'.")
        return []

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
        tags = [category_name]  # if tags not available on geeksforgeeks Use the last part of the URL as a tag
    
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


# base_url = 'https://www.geeksforgeeks.org/fundamentals-of-algorithms/'
# algorithms = scrape_all_algorithms(base_url)
# save_to_json(algorithms)

# script data from a page 


def scrape_gfg_sliding_window():
    """Scrape Window Sliding Technique page from GeeksforGeeks."""
    url = "https://www.geeksforgeeks.org/window-sliding-technique/"
    
    # Send request to the webpage
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.geeksforgeeks.org/',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None
    
    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Get the main content
    main_content = soup.find('article', class_='article-page')
    if not main_content:
        print("Could not find main content.")
        return None
    
    # Extract title
    title = soup.find('h1', class_='title')
    title_text = title.text.strip() if title else "Window Sliding Technique"
    
    # Extract main description
    description = ""
    content_div = main_content.find('div', class_='text_content')
    if content_div:
        # Get all paragraphs directly under the content div (not in other divs)
        paragraphs = content_div.find_all('p', recursive=False)
        description = '\n'.join([p.text.strip() for p in paragraphs])
    
    # Extract code examples
    code_blocks = []
    code_divs = main_content.find_all('div', class_='code-block')
    for code_div in code_divs:
        code_text = code_div.text.strip()
        language = "unknown"
        
        # Try to determine the language
        if 'class="language-' in str(code_div):
            lang_match = re.search(r'class="language-(\w+)"', str(code_div))
            if lang_match:
                language = lang_match.group(1)
        
        # Look for common language indicators in the code
        if 'def ' in code_text and ('return' in code_text or 'print(' in code_text):
            language = "python"
        elif 'public static void main' in code_text:
            language = "java"
        elif '#include' in code_text and ('{' in code_text and '}' in code_text):
            language = "cpp"
        
        code_blocks.append({
            "language": language,
            "code": code_text
        })
    
    # Extract sidebar content
    sidebar_data = {}
    sidebar = soup.find('div', class_='leftBarText')
    if sidebar:
        # Find all divs that might contain topic sections
        topic_divs = sidebar.find_all('div', recursive=False)
        
        for div in topic_divs:
            # Get the heading of the section
            heading = div.find(['h2', 'h3', 'h4'])
            if heading:
                section_title = heading.text.strip()
                
                # Get links in this section
                links = div.find_all('a')
                
                if links:
                    sidebar_data[section_title] = []
                    for link in links:
                        sidebar_data[section_title].append({
                            "title": link.text.strip(),
                            "url": link.get('href', '')
                        })
    
    # Extract complexity information
    complexity = {"time": "", "space": ""}
    complexity_section = main_content.find(lambda tag: tag.name == 'h2' and 'Time Complexity' in tag.text)
    if complexity_section:
        # Get the next paragraph after Time Complexity heading
        next_p = complexity_section.find_next('p')
        if next_p:
            complexity["time"] = next_p.text.strip()
    
    space_section = main_content.find(lambda tag: tag.name == 'h2' and 'Space Complexity' in tag.text)
    if space_section:
        next_p = space_section.find_next('p')
        if next_p:
            complexity["space"] = next_p.text.strip()
    
    # Extract related problems
    related_problems = []
    related_section = main_content.find(lambda tag: tag.name in ['h2', 'h3'] and 'Related Problems' in tag.text)
    if related_section:
        # Get all links after the Related Problems heading
        ul = related_section.find_next('ul')
        if ul:
            links = ul.find_all('a')
            for link in links:
                related_problems.append({
                    "title": link.text.strip(),
                    "url": link.get('href', '')
                })
    
    # Construct the final data structure
    algorithm_data = {
        "title": title_text,
        "description": description,
        "code": code_blocks,
        "complexity": complexity,
        "sidebar_content": sidebar_data,
        "related_problems": related_problems,
        "url": url
    }
    
    return algorithm_data

def save_to_json(data, filename="sliding_window_technique.json"):
    """Save the scraped data to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to file: {e}")

def main():
    print("Scraping GeeksforGeeks Window Sliding Technique page...")
    data = scrape_gfg_sliding_window()
    
    if data:
        save_to_json(data)
        print("Scraping completed successfully!")
    else:
        print("Failed to scrape the page.")

if __name__ == "__main__":
    main()