import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
from tqdm import tqdm
from .models import Thesaurus, Translations


class ThesaurusScraper:
    def __init__(self):
        self.base_url = "https://vocabularies.unesco.org/browser"
        self.alphabet = "ABCDEFGHIJKLMNOUPQRSTUVWXYZ"
        self.vowels_with_accents = ["Á", "É", "Í", "Ó", "Ú"]
        self.timeout = 15

    def scrape(self):
        """
        # Loop through the alphabet and get the results for each letter
        for letter in tqdm(self.alphabet, desc="Processing letters"):
            self.get_results(letter)

        # Loop through the vowels with accents and get the results for each vowel
        for vowel in tqdm(self.vowels_with_accents, desc="Processing vowels"):
            self.get_results(urllib.parse.quote(vowel))
        """
        
        results_2_detail = Thesaurus.objects.exclude(translations__isnull=False)
        for result in tqdm(results_2_detail, desc="Getting details"):
            self.get_details(result)

    def get_results(self, letter):
        results = []
        offset = 0
        url = f"{self.base_url}/thesaurus/en/index/{letter}?offset={offset}&clang=en"
        try:
            response = requests.get(url, timeout=self.timeout)
        except Exception:
            return results

        # Loop through the results pages
        while response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Find the list of results
            try:
                ul = soup.find("ul", {"class": "alphabetical-search-results"})
                lis = ul.find_all("li")
            except AttributeError:
                break

            # Loop through the results and append to the results list
            for li in lis:
                a = li.find("a")
                if a:
                    link = a["href"]
                    name = a.text.strip()
                    Thesaurus.objects.update_or_create(
                        name=name.strip(), defaults={"link": link}
                    )
                replaced = li.find("span")
                if replaced:
                    name = a.text.strip()
                    Thesaurus.objects.update_or_create(
                        name=name.strip(), defaults={"deprecated": True}
                    )

            # Increment the offset and get the next page of results
            offset += 250
            url = (
                f"{self.base_url}/thesaurus/en/index/{letter}?offset={offset}&clang=en"
            )

            try:
                response = requests.get(url, timeout=self.timeout)
            except Exception:
                break

        return results

    def get_details(self, result):
        link = result.link
        if link and link[0] == "/":
            link = link[1:]

        url = f"{self.base_url}/{link}"
        try:
            response = requests.get(url, timeout=self.timeout)
        except Exception:
            return

        if response.status_code != 200:
            return

        soup = BeautifulSoup(response.content, "html.parser")

        # Find the related concepts
        related_concepts = soup.find("span", title="Concepts related to this concept.")
        if related_concepts:
            ul = related_concepts.parent.find_next_sibling("div").find("ul")
            for li in ul.find_all("li"):
                a = li.find("a")
                if a:
                    name = a.text.strip()
                    link = a["href"]
                    related, _ = Thesaurus.objects.update_or_create(
                        name=name, defaults={"link": link}
                    )
                    result.related_thesauri.add(related)

        # Find the parent thesaurus
        broader_concept = soup.find("span", title="Broader concept")
        if broader_concept:
            ul = broader_concept.parent.find_next_sibling("div").find("ul")
            for li in ul.find_all("li"):
                a = li.find("a")
                if a:
                    name = a.text.strip()
                    link = a["href"]
                    parent, _ = Thesaurus.objects.update_or_create(
                        name=name.strip(), defaults={"link": link}
                    )
                    result.parent_thesaurus = parent
                    result.save()

        # Find the translations in other languages
        other_languages = soup.find_all("a", hreflang=True, class_=False)
        if other_languages:
            for a in other_languages:
                language = a["hreflang"]
                name = a.text.strip()
                try:
                    Translations.objects.get(language=language, name=name)
                except Translations.DoesNotExist:
                    Translations.objects.create(language=language, name=name, thesaurus=result)


def start_scraping():
    scraper = ThesaurusScraper()
    scraper.scrape()
