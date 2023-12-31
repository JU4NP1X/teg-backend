import requests
import os
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
from unidecode import unidecode
import hunspell
from django.db.models import Q
from categories.models import (
    Categories,
    Translations,
    Authorities,
    update_categories_tree,
)

BASE_DIR = os.path.dirname(os.path.realpath(__name__))

dic = hunspell.HunSpell(
    f"{BASE_DIR}/dictionary/es_ANY.dic", f"{BASE_DIR}/dictionary/es_ANY.aff"
)

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
            .filter(deprecated=False)
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
                    defaults={"link": link},
                )

    def get_details(self, result):
        """
        Get the details for a given result.

        Args:
            result (Categories): The result to get the details for.
        """

        for lang in ["es", "en"]:
            url = f"{self.base_url}/{lang}/{result.link}"

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
                        name=name,
                        authority=self.authority,
                    ).first()
                    if parent and parent.deprecated:
                        update_categories_tree(result, parent.parent)
                    else:
                        update_categories_tree(result, parent)

                deprecated = soup.find("b", text="USE")
                if deprecated:
                    Categories.objects.filter(pk=result.id).update(deprecated=True)
                else:
                    Categories.objects.filter(pk=result.id).update(deprecated=False)
            else:
                trans_key = soup.find("b", text="PC")
                if trans_key:
                    translation = (
                        trans_key.find_parent()
                        .text.replace("PC:", "")
                        .strip()
                        .capitalize()
                    )

                    if translation:
                        if len(translation) < 4:
                            trans_key = soup.find("b", text="NA")
                            if trans_key:
                                translation = soup.get_text().strip()
                                translation = re.search(
                                    r"NA: (.+?)\n", translation
                                ).group(1)
                                translation = translation.strip(".").lower()

                        words = translation.split()

                        correction = ""
                        for word in words:
                            if correction == "":
                                suggestions = dic.suggest(word.capitalize())
                            else:
                                suggestions = dic.suggest(word.lower())
                            finded = False
                            if len(suggestions):
                                for suggestion in suggestions:
                                    if unidecode(suggestion).lower() == word.lower():
                                        finded = True
                                        break
                            if not finded:
                                if len(word) > 1 and len(word) < 7:
                                    suggestion = word.upper()
                                else:
                                    suggestion = word.lower()

                            if correction == "":
                                suggestion = suggestion[0].upper() + suggestion[1:]

                            correction = correction + " " + suggestion

                        correction = correction.strip()

                        if len(correction):
                            Translations.objects.update_or_create(
                                language="es",
                                category=result,
                                defaults={"name": correction},
                            )
                        else:
                            Translations.objects.update_or_create(
                                language="es",
                                category=result,
                                defaults={"name": translation.strip()},
                            )
