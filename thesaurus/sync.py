import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from .models import Thesaurus
from tqdm import tqdm


class ThesaurusScraper:
    def __init__(self):
        self.base_url = 'https://vocabularies.unesco.org/browser'
        self.alphabet = 'ABCDEFGHIJKLMNOUPQRSTUVWXYZ'
        self.vowels_with_accents = ['Á', 'É', 'Í', 'Ó', 'Ú']
        self.timeout = 15

    def scrape(self):
        """
        # Loop through the alphabet and get the results for each letter
        for letter in tqdm(self.alphabet, desc='Processing letters'):
            results = self.get_results(letter)
            Thesaurus.objects.bulk_create(results, ignore_conflicts=True)

        # Loop through the vowels with accents and get the results for each vowel
        for vowel in tqdm(self.vowels_with_accents, desc='Processing vowels'):
            results = self.get_results(urllib.parse.quote(vowel))
            Thesaurus.objects.bulk_create(results, ignore_conflicts=True)
        """
        #results_2_detail = Thesaurus.objects.exclude(translations__isnull=False)
        results_2_detail = Thesaurus.objects.exclude(parent_thesaurus__isnull=False)
        for result in tqdm(results_2_detail, desc='Getting details'):
            self.get_details(result)

    def get_results(self, letter):
        results = []
        offset = 0
        url = f'{self.base_url}/thesaurus/en/index/{letter}?offset={offset}&clang=en'
        try:
            response = requests.get(url, timeout=self.timeout)
        except Timeout:
            return results
        
        # Loop through the results pages
        while response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the list of results
            try:
                ul = soup.find('ul', {'class': 'alphabetical-search-results'})
                lis = ul.find_all('li')
            except AttributeError:
                break

            # Loop through the results and append to the results list
            for li in lis:
                a = li.find('a')
                if a:
                    link = a['href']
                    name = a.text.strip()
                    results.append(Thesaurus(link=link, name=name))
                replaced = li.find('span')
                if replaced:
                    name = a.text.strip()
                    results.append(Thesaurus(deprecated=True, name=name))

            # Increment the offset and get the next page of results
            offset += 250
            url = f'{self.base_url}/thesaurus/en/index/{letter}?offset={offset}&clang=en'
            
            try:
                response = requests.get(url, timeout=self.timeout)
            except Timeout:
                break

        return results

    def get_details(self, result):
        link = result.link
        if link and link[0] == '/':
            link = link[1:]

        url = f'{self.base_url}/{link}'
        try:
            response = requests.get(url, timeout=self.timeout)
        except Timeout:
            return

        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the related concepts
        related_concepts = soup.find('span', title='Concepts related to this concept.')
        if related_concepts:
            ul = related_concepts.parent.find_next_sibling('div').find('ul')
            for li in ul.find_all('li'):
                a = li.find('a')
                if a:
                    name = a.text.strip()
                    link = a['href']
                    related = Thesaurus.objects.get_or_create(link=link, name=name)[0]
                    result.related_thesauri.add(related)

        # Find the parent thesaurus
        broader_concept = soup.find('span', title='Broader concept')
        if broader_concept:
            ul = broader_concept.parent.find_next_sibling('div').find('ul')
            for li in ul.find_all('li'):
                a = li.find('a')
                if a:
                    name = a.text.strip()
                    link = a['href']
                    parent = Thesaurus.objects.get_or_create(link=link, name=name.strip())[0]
                    result.parent_thesaurus = parent
                    result.save()

        # Find the translations in other languages
        other_languages = soup.find_all('a', hreflang=True, class_=False)
        if other_languages:
            for a in other_languages:
                language = a['hreflang']
                name = a.text.strip()
                result.translations.get_or_create(language=language, name=name)

                
def start_scraping():
    scraper = ThesaurusScraper()
    scraper.scrape()