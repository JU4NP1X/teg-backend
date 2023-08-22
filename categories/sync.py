import requests
import urllib.parse
from bs4 import BeautifulSoup
from tqdm import tqdm
from .models import Categories, Translations


class CategoriesScraper:
    """
    Class for scraping data from a website.

    Attributes:
        base_url (str): The base URL of the website.
        timeout (int): The timeout for the requests.
        alphabet (str): The alphabet to iterate through.
        vowels_with_accents (list): The list of vowels with accents.
    """

    def __init__(self):
        self.base_url = "https://vocabularies.unesco.org/browser"
        self.alphabet = "ABCDEFGHIJKLMNOUPQRSTUVWXYZ"
        self.vowels_with_accents = ["Á", "É", "Í", "Ó", "Ú"]
        self.timeout = 15

    def scrape(self):
        """
        Scrape data from the website.

        This method iterates through the alphabet and vowels with accents,
        and calls the corresponding methods to get the results and details.

        """
        # Loop through the alphabet and get the results for each letter
        for letter in tqdm(self.alphabet, desc="Processing letters"):
            self.get_results(letter)

        # Loop through the vowels with accents and get the results for each vowel
        for vowel in tqdm(self.vowels_with_accents, desc="Processing vowels"):
            self.get_results(urllib.parse.quote(vowel))

        results_2_detail = Categories.objects.exclude(translations__isnull=False)
        for result in tqdm(results_2_detail, desc="Getting details"):
            self.get_details(result)

    def get_results(self, letter):
        """
        Get the results for a given letter.

        Args:
            letter (str): The letter to get the results for.

        Returns:
            list: The list of results.
        """
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
                    Categories.objects.update_or_create(
                        name=name, defaults={"link": link}
                    )
                replaced = li.find("span")
                if replaced:
                    name = a.text.strip()
                    Categories.objects.update_or_create(
                        name=name, defaults={"deprecated": True}
                    )

            # Increment the offset and get the next page of results
            offset += 250
            url = (
                f"{self.base_url}/categories/en/index/{letter}?offset={offset}&clang=en"
            )

            try:
                response = requests.get(url, timeout=self.timeout)
            except Exception:
                break

        return results

    def get_details(self, result):
        """
        Get the details for a given result.

        Args:
            result (Categories): The result to get the details for.
        """
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
                    related, _ = Categories.objects.update_or_create(
                        name=name, defaults={"link": link}
                    )
                    result.related_categories.add(related)

        # Find the parent categories
        broader_concept = soup.find("span", title="Broader concept")
        if broader_concept:
            ul = broader_concept.parent.find_next_sibling("div").find("ul")
            for li in ul.find_all("li"):
                a = li.find("a")
                if a:
                    name = a.text.strip()
                    link = a["href"]
                    parent, _ = Categories.objects.update_or_create(
                        name=name.strip(), defaults={"link": link}
                    )
                    result.parent_category = parent
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
                    Translations.objects.create(
                        language=language, name=name, categories=result
                    )


def start_scraping():
    """
    Starts the scraping proccess
    """
    scraper = CategoriesScraper()
    scraper.scrape()
