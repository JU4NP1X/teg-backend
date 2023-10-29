import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from django.db.models import Q
from categories.models import (
    Categories,
    Translations,
    Authorities,
    categories_tree_adjust,
    update_categories_tree,
)

requests.packages.urllib3.disable_warnings()


class UnescoScraper:
    """
    Class for scraping data from a website.

    Attributes:
        base_url (str): The base URL of the website.
        timeout (int): The timeout for the requests.
        alphabet (str): The alphabet to iterate through.
    """

    def __init__(self):
        self.base_url = "https://vocabularies.unesco.org/browser"
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.timeout = 15
        self.authority = Authorities.objects.get(name="UNESCO")

    def scrape(self):
        """
        Scrape data from the website.

        This method iterates through the alphabet and vowels with accents,
        and calls the corresponding methods to get the results and details.

        """
        # Loop through the alphabet and get the results for each letter
        for letter in tqdm(self.alphabet, desc="Processing letters"):
            self.get_results(letter)

        results_2_detail = (
            Categories.objects.filter(authority=self.authority)
            .filter(translations=None, deprecated=False)
            .exclude(link="")
        )
        for result in tqdm(results_2_detail, desc="Getting details"):
            self.get_details(result)

        # Break the relationship between deprecated categories and their children
        deprecated_categories = Categories.objects.filter(
            Q(parent__isnull=False) | Q(children__isnull=False),
            deprecated=True,
        )
        for category in deprecated_categories:
            descendants = Categories.objects.filter(parent__id=category.id)
            current_category = Categories.objects.get(id=category.id)
            for descendant in descendants:
                current_descendant = Categories.objects.get(id=descendant.id)
                update_categories_tree(current_descendant, current_category.parent)
            current_category = Categories.objects.get(id=category.id)
            update_categories_tree(current_category, None)

        categories_tree_adjust()

    def get_results(self, letter):
        """
        Get the results for a given letter.

        Args:
            letter (str): The letter to get the results for.

        Returns:
            list: The list of results.
        """
        offset = 0
        url = f"{self.base_url}/thesaurus/en/index/{letter}?offset={offset}&clang=en"
        try:
            response = requests.get(url, timeout=self.timeout, verify=False)
        except Exception as e:
            print(e)
            return
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
                    category, _ = Categories.objects.update_or_create(
                        name=name,
                        authority=self.authority,
                        defaults={"link": link, "deprecated": False},
                    )
                    replaced = li.find("span")
                    if replaced:
                        name = replaced.text.strip()
                        Categories.objects.update_or_create(
                            name=replaced.text.strip(),
                            authority=self.authority,
                            deprecated=True,
                        )
                    category.save()

            # Increment the offset and get the next page of results
            offset += 250
            url = (
                f"{self.base_url}/categories/en/index/{letter}?offset={offset}&clang=en"
            )

            try:
                response = requests.get(url, timeout=self.timeout, verify=False)
            except Exception:
                break

    def get_details(self, result):
        """
        Get the details for a given result.

        Args:
            result (Categories): The result to get the details for.
        """
        link = result.link
        if link == "":
            return
        if link and link[0] == "/":
            link = link[1:]

        url = f"{self.base_url}/{link}"
        try:
            response = requests.get(url, timeout=self.timeout, verify=False)
        except Exception as exept:
            print(exept)
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
                        name=name, authority=self.authority, defaults={"link": link}
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
                        name=name.strip(),
                        authority=self.authority,
                        defaults={"link": link},
                    )
                    update_categories_tree(result, parent)

        # Find the translations in other languages
        other_languages = soup.find_all("a", hreflang=True, class_=False)
        if other_languages:
            for a in other_languages:
                language = a["hreflang"]
                name = a.text.strip()
                Translations.objects.update_or_create(
                    language=language,
                    category=result,
                    defaults={"name": name},
                )
