import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from django.db.models import Q
from .models import Categories, Translations, Authorities
from django.db import connection

requests.packages.urllib3.disable_warnings()


def execute_query(query):
    with connection.cursor() as cursor:
        cursor.execute(query)


def categories_tree_adjust():
    query = """
        UPDATE categories AS ca
        SET tree_id = ca2.tree_id, "level" = ca2.level + 1
        FROM categories AS ca2
        WHERE ca2.tree_id <> ca.tree_id AND ca2.id = ca.parent_id
    """

    # Tree correction (there is a bug, that the move_to not update the tree of it chindrens)
    while True:
        execute_query(query)
        rows_affected = connection.cursor().rowcount
        if rows_affected <= 0:
            break

    query = """
        UPDATE categories AS ca
        SET "level" = 0
        WHERE ca.deprecated = TRUE
    """

    execute_query(query)


class CategoriesScraper:
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
            .filter(translations=None)
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
                if current_category.parent == None:
                    current_category.move_to(None, position="right")
                else:
                    current_descendant.parent = current_category.parent
                current_descendant.save()
            current_category = Categories.objects.get(id=category.id)
            current_category.move_to(None, position="right")
            current_category.save()

        categories_tree_adjust()

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
            response = requests.get(url, timeout=self.timeout, verify=False)
        except Exception as e:
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
                    category, _ = Categories.objects.update_or_create(
                        name=name,
                        authority=self.authority,
                        deprecated=False,
                        defaults={"link": link},
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

        return results

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
                    if (not parent.deprecated) and (not result.deprecated):
                        result.parent = parent
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
                        language=language, name=name, category=result
                    )


def start_scraping():
    """
    Starts the scraping process
    """
    scraper = CategoriesScraper()
    scraper.scrape()
