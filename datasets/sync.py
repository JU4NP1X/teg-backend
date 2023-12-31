"""
Class for scraping datasets based on category and universities.
"""
import os
import re
import time
import requests
import subprocess
from langdetect import detect
from tqdm import tqdm
from googletrans import Translator as GoogleTranslator
from django.db.models import Q
from categories.models import Categories
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from .models import Datasets, DatasetsEnglishTranslations, DatasetsUniversity

requests.packages.urllib3.disable_warnings()


class OneSearchScraper:
    """
    Class for scraping datasets based on category and universities.

    Attributes:
        category (Categories): The category of the datasets.
        universities (list): List of universities to scrape datasets from.
        timeout (int): Timeout for the HTTP requests.
        session (requests.Session): Session object for making HTTP requests.
        current_university (university): The current university being scraped.
    """

    def __init__(self, category, universities):
        self.category = category
        self.universities = universities
        self.timeout = 40
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
        )
        self.current_university = None

    def get_base_url(self, university):
        """
        Get the base URL for a given university.

        Args:
            university (str): The name of the university.

        Returns:
            str: The base URL of the university.
        """
        university_obj = DatasetsUniversity.objects.filter(name=university).first()
        if university_obj:
            self.current_university = university_obj
            return university_obj.url
        return None

    def scrape(self):
        """
        Scrape datasets from the specified universities.

        This method retrieves datasets from the universities' API based on the specified category.
        """
        for university in self.universities:
            base_url = self.get_base_url(university)
            if not base_url:
                continue

            base_url += "/primaws/rest/pub/pnxs"

            start = 0
            while start <= 50:
                try:
                    params = {
                        "blendFacetsSeparately": "false",
                        "disableCache": "false",
                        "getMore": "0",
                        "inst": "61UWA_INST",
                        "lang": "en",
                        "limit": "50",
                        "mode": "advanced",
                        "newspapersActive": "false",
                        "newspapersSearch": "false",
                        "offset": start,
                        "pcAvailability": "false",
                        "q": "sub,exact,{}".format(self.category.name),
                        "qExclude": "",
                        "qInclude": "",
                        "rapido": "false",
                        "refEntryActive": "false",
                        "rtaLinks": "true",
                        "scope": "MyInst_and_CI",
                        "searchInFulltextUserSelection": "false",
                        "skipDelivery": "Y",
                        "sort": "rank",
                        "tab": "Everything",
                        "vid": self.current_university.vid,
                    }
                    response = self.session.get(
                        base_url, params=params, timeout=self.timeout, verify=False
                    )
                    response.raise_for_status()
                    data = response.json()

                    if not data["docs"]:
                        break

                    for doc in data["docs"]:
                        try:
                            title = doc["pnx"]["display"]["title"][0].strip()
                            description = doc["pnx"]["addata"]["abstract"][0].strip()
                            categories = []
                            for category in doc["pnx"]["display"]["subject"]:
                                categories.extend(category.split(";"))
                        except KeyError:
                            continue

                        # Check if the document has at least one categories associated with its subject
                        if not categories:
                            continue

                        dataset_categories = []
                        for category in categories:
                            # Filter thesauri that meet the criteria for a similar search
                            categories = Categories.objects.filter(
                                name__icontains=category,
                                authority__id=self.category.authority_id,
                            ).first()
                            if categories:
                                dataset_categories.append(categories)
                            else:
                                categories = Categories.objects.filter(
                                    Q(name__icontains=category)
                                    | Q(name__icontains="category"),
                                    authority__id=self.category.authority_id,
                                ).first()
                                if categories:
                                    dataset_categories.append(categories)

                        # Check if the dataset has at least one associated categories before creating it.
                        if not dataset_categories:
                            continue

                        if self.category.name.lower() in description.lower():
                            dataset_categories.append(self.category)

                        dataset_categories = list(set(dataset_categories))
                        dataset = Datasets.objects.filter(paper_name=title).first()

                        if not dataset and len(title) <= 250 and len(description) > 1:
                            dataset = Datasets.objects.create(
                                paper_name=title,
                                summary=description,
                                university=self.current_university,
                            )
                        elif dataset and len(description) > len(dataset.summary):
                            dataset.summary = description
                            dataset.save()

                        if dataset:
                            # Retrieve the existing categories of the dataset.
                            existing_categories = dataset.categories.all()
                            for category_obj in dataset_categories:
                                # Add the new category to the existing set.
                                if category_obj not in existing_categories:
                                    dataset.categories.add(category_obj)

                    start += 50
                except Exception as e:
                    print(f"Error getting data: {e}")


import random
import time


class GoogleScholarScraper:
    def __init__(self):
        self.current_university = DatasetsUniversity.objects.filter(
            name="google_scholar"
        ).first()
        self.headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://scholar.google.com/",
            "Connection": "keep-alive",
        }

    def search_and_save_datasets(self, category):
        page = 0
        count = 0  # Counter for the number of datasets found
        query = category.name

        while count < 20:
            if page == 0:
                search_url = f"{self.current_university.url}/scholar?q={query}"
            else:
                search_url = (
                    f"{self.current_university.url}/scholar?q={query}&start={page * 10}"
                )

            user_agent = UserAgent().random
            self.headers["User-Agent"] = user_agent
            try:
                response = requests.get(
                    search_url, headers=self.headers, timeout=40, verify=False
                )
                response.raise_for_status()

                if response.status_code == 429:
                    # Si se recibe un error 429, esperar un tiempo aleatorio antes de realizar la siguiente solicitud
                    wait_time = random.uniform(1, 3)
                    time.sleep(wait_time)
                    continue

                soup = BeautifulSoup(response.content, "html.parser")
                result_links = soup.select(".gs_rt a")

                for link in result_links:
                    if count >= 20:
                        break

                    url = link["href"]
                    try:
                        title, meta_description = self.get_page_info(url)
                        title = self.clean_data(title)
                        meta_description = self.clean_data(meta_description)
                        if len(meta_description) >= 250:
                            dataset = Datasets.objects.filter(paper_name=title).first()

                            if not dataset and len(title) <= 250:
                                dataset = Datasets.objects.create(
                                    paper_name=title,
                                    summary=meta_description,
                                    university=self.current_university,
                                )
                            if dataset:
                                dataset.save()
                                existing_categories = dataset.categories.all()
                                if category not in existing_categories:
                                    dataset.categories.add(category)
                                count += 1
                    except Exception:
                        print(f"Error retrieving page info for URL: {url}")

                page += 1
                time.sleep(2)
            except Exception as e:
                print(f"HTTPError: {e}")
                # Si se recibe un error HTTP, esperar un tiempo aleatorio antes de realizar la siguiente solicitud
                wait_time = random.uniform(1, 3)
                time.sleep(wait_time)
                break

    def get_page_info(self, url):
        try:
            user_agent = UserAgent().random
            self.headers["User-Agent"] = user_agent
            response = requests.get(url, headers=self.headers, timeout=40, verify=False)
            response.raise_for_status()

            if response.status_code == 429:
                # Si se recibe un error 429, esperar un tiempo aleatorio antes de realizar la siguiente solicitud
                wait_time = random.uniform(1, 3)
                time.sleep(wait_time)
                return "", ""

            soup = BeautifulSoup(response.content, "html.parser")
            title_tag = soup.find("title")
            meta_tag = soup.find("meta", attrs={"name": "description"})

            title = title_tag.text if title_tag else ""
            meta_description = meta_tag["content"] if meta_tag else ""

            return title, meta_description
        except requests.exceptions.HTTPError as e:
            print(f"HTTPError: {e}")
            return "", ""

    def clean_data(self, data):
        # Remove invisible characters except spaces and tabs
        cleaned_data = re.sub(r"[^\S\r\n\t ]", "", data)
        return cleaned_data


class CreateMissingTranslations:
    @staticmethod
    def translate_all():
        """
        Create missing translations for datasets.

        This method creates English translations for datasets that don't have them.
        """
        datasets = Datasets.objects.exclude(datasetsenglishtranslations__isnull=False)
        for dataset in tqdm(datasets, desc="Creating translations"):
            try:
                translation = DatasetsEnglishTranslations(
                    dataset=dataset,
                    paper_name=CreateMissingTranslations.translate_text(
                        dataset.paper_name
                    ),
                    summary=CreateMissingTranslations.translate_text(dataset.summary),
                )
                if translation.paper_name != "" and dataset.summary != "":
                    translation.save()
            except Exception as error:
                print(f"Error translating text: {error}")
                print(dataset.paper_name + " " + dataset.summary)

    @staticmethod
    def pass_english_text():
        """
        Pass English text datasets.

        This method checks if the dataset's paper name and summary are in English and saves them as English translations.
        """
        datasets = Datasets.objects.exclude(datasetsenglishtranslations__isnull=False)
        for dataset in tqdm(datasets, desc="Parsing English text"):
            try:
                name_alpha2 = detect(dataset.paper_name)
                summary_alpha2 = detect(dataset.summary)
                if name_alpha2 == "en" and summary_alpha2 == "en":
                    translation = DatasetsEnglishTranslations(
                        dataset=dataset,
                        paper_name=dataset.paper_name,
                        summary=dataset.summary,
                    )
                    if translation.paper_name != "" and dataset.summary != "":
                        translation.save()
            except Exception as e:
                print(f"Error translating text: {e}")
                print(dataset.paper_name + " " + dataset.summary)

    @staticmethod
    def translate_text(title):
        """
        Translate text to English.

        This method translates the given text to English using Google Translate API.

        Args:
            title (str): The text to be translated.

        Returns:
            str: The translated text.
        """
        origin_alpha2 = detect(title)

        if origin_alpha2 != "en":
            translator = GoogleTranslator()
            try:
                translated_text = translator.translate(title[:500], dest="en").text
            except Exception:
                translated_text = ""
        else:
            translated_text = title
        return translated_text


def start_scraping():
    subprocess.Popen(
        [
            os.getenv("PYTHON_PATH"),
            "./manage.py",
            "datasets_sync",
        ]
    )
