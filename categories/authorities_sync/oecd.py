import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from django.db.models import Q
from categories.models import (
    Categories,
    Translations,
    Authorities,
    update_categories_tree,
)
from django.db import connection

requests.packages.urllib3.disable_warnings()


class OecdScraper:
    """
    Class for scraping data from a website.

    Attributes:
        base_url (str): The base URL of the website.
        timeout (int): The timeout for the requests.
        alphabet (str): The alphabet to iterate through.
    """

    def __init__(self):
        self.base_url = "https://bibliotecavirtual.clacso.org.ar/ar/oecd-macroth"
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.timeout = 15
        self.authority = Authorities.objects.get(name="OECD")

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
            .filter(parent=None, deprecated=False)
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
        url = f"{self.base_url}/en/{letter}.html"
        try:
            response = requests.get(url, timeout=self.timeout, verify=False)
        except Exception as e:
            print(e)
            return

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")

            ol = soup.find("ol")
            aTags = ol.find_all("a")

            # Loop through the results and append to the results list
            for a in aTags:
                link = a["href"]
                name = a.text.strip().capitalize()
                Categories.objects.update_or_create(
                    name=name,
                    authority=self.authority,
                    defaults={"link": link, "deprecated": False},
                )

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

        for lang in ["en", "es"]:
            url = f"{self.base_url}/{lang}/{link}"

            try:
                response = requests.get(url, timeout=self.timeout, verify=False)
            except Exception as exept:
                print(exept)
                return

            if response.status_code != 200:
                return

            soup = BeautifulSoup(response.content, "html.parser")

            # Find the related concepts
            if lang == "en":
                related_concepts = soup.find("b", text="RT")
                while related_concepts:
                    a = related_concepts.find_next_sibling("a")
                    name = a.text.strip().capitalize()
                    link = a["href"]
                    related, _ = Categories.objects.update_or_create(
                        name=name, authority=self.authority, defaults={"link": link}
                    )
                    result.related_categories.add(related)
                    result.save()
                    related_concepts = related_concepts.find_next_sibling(
                        "b", text="RT"
                    )

                # Find the parent categories
                broader_concept = soup.find("b", text="BT")
                if broader_concept:
                    a = broader_concept.find_next_sibling("a")
                    name = a.text.strip().capitalize()
                    link = a["href"]
                    parent = Categories.objects.filter(
                        name__icontains=name,
                        authority=self.authority,
                    ).first()
                    update_categories_tree(result, parent)
            else:
                trans_key = soup.find("b", text="PC")
                translation = (
                    trans_key.find_next_sibling().strip(":").strip().capitalize()
                )
                Translations.objects.update_or_create(
                    language="es",
                    category=result,
                    defaults={"name": translation},
                )