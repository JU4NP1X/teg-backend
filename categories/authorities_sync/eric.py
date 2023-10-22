import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from googletrans import Translator as GoogleTranslator
from categories.models import (
    Categories,
    Translations,
    Authorities,
    update_categories_tree,
)
from django.db import connection

requests.packages.urllib3.disable_warnings()
translator = GoogleTranslator()


class EricScraper:
    """
    Class for scraping data from a website.

    Attributes:
        base_url (str): The base URL of the website.
        timeout (int): The timeout for the requests.
        alphabet (str): The alphabet to iterate through.
    """

    def __init__(self):
        self.base_url = "https://eric.ed.gov"
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.timeout = 15
        self.authority = Authorities.objects.get(name="ERIC")

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
            Categories.objects.filter(
                authority=self.authority, deprecated=False
            ).filter(parent=None)
            # .filter(link="")
            .exclude(link="")
        )
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
        url = f"{self.base_url}?ti={letter}"
        try:
            response = requests.get(url, timeout=self.timeout, verify=False)
        except Exception as e:
            print(e)
            return

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            # Find the list of results in col1, col2, and col3
            for col in ["col1", "col2", "col3"]:
                try:
                    div = soup.find("div", {"class": col})
                    ul = div.find("ul")
                    lis = ul.find_all("li")
                except AttributeError:
                    continue

                # Loop through the results and append to the results list
                for li in lis:
                    a = li.find("a")
                    if a:
                        link = a["href"]
                        name = a.text.strip()
                        deprecated = False
                        if li.find("em"):
                            deprecated = True
                        if not deprecated:
                            print(name)
                        Categories.objects.update_or_create(
                            name=name,
                            authority=self.authority,
                            defaults={"link": link, "deprecated": deprecated},
                        )

    def get_details(self, result):
        """
        Get the details for a given result.

        Args:
            result (Categories): The result to get the details for.
        """
        link = result.link

        if not len(Translations.objects.filter(category_id=result.id, language="es")):
            try:
                translation = translator.translate(result.name, dest="es").text
                translation_object = Translations.objects.get_or_create(
                    language="es",
                    category=result,
                    defaults={"name": translation},
                )
                print(translation_object)
            except Exception as exept:
                print(exept)

        if not link:
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
            related_concepts = soup.find("div", text="Related Terms")
            if related_concepts:
                for a in related_concepts.find_next_siblings("a"):
                    name = a.text.strip()
                    link = a["href"]
                    related, _ = Categories.objects.update_or_create(
                        name=name, authority=self.authority, defaults={"link": link}
                    )
                    result.related_categories.add(related)
                    result.save()

            # Find the parent categories
            broader_concept = soup.find("div", text="Broader Terms")
            if broader_concept:
                a = broader_concept.find_next_sibling()
                if a.name == "a":
                    name = a.text.strip()
                    link = a["href"]
                    parent = Categories.objects.filter(
                        name=name,
                        authority=self.authority,
                    ).first()
                    update_categories_tree(result, parent)
                    return
        update_categories_tree(result, None)
