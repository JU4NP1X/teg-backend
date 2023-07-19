import requests
from requests.exceptions import Timeout
from thesaurus.models import Thesaurus
from .models import Datasets, Datasets_English_Translations, Datasets_University
from translate import Translator
from langdetect import detect
from tqdm import tqdm
from googletrans import Translator as GoogleTranslator


class DatasetsScraper:
    def __init__(self, query, universities):
        self.query = query
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
        university_obj = Datasets_University.objects.filter(name=university).first()
        if university_obj:
            self.current_university = university_obj
            return university_obj.url
        return None

    def scrape(self):
        for university in self.universities:
            base_url = self.get_base_url(university)
            if not base_url:
                continue

            base_url += "/primaws/rest/pub/pnxs"

            start = 0
            while start < 500:
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
                        "q": "sub,exact,{}".format(self.query),
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
                        base_url, params=params, timeout=self.timeout
                    )
                    response.raise_for_status()
                    data = response.json()

                    if not data["docs"]:
                        break

                    for doc in data["docs"]:
                        try:
                            title = doc["pnx"]["display"]["title"][0]
                            description = doc["pnx"]["addata"]["abstract"][0]
                            categories = []
                            for category in doc["pnx"]["display"]["subject"]:
                                categories.extend(category.split(";"))
                        except KeyError:
                            continue

                        # Check if the document has at least one thesaurus associated with its subject
                        if not categories:
                            continue

                        dataset_categories = []
                        for category in categories:
                            # Filter thesauri that meet the criteria for a similar search
                            thesaurus = Thesaurus.objects.filter(
                                name__icontains=category
                            ).first()
                            if thesaurus:
                                dataset_categories.append(thesaurus)

                        # Check if the dataset has at least one associated thesaurus before creating it.
                        if not dataset_categories:
                            continue
                        dataset = Datasets.objects.filter(paper_name=title).first()

                        if not dataset:
                            dataset = Datasets.objects.create(
                                paper_name=title,
                                summary=description,
                                university=self.current_university,
                            )
                        elif len(description) > len(dataset.summary):
                            dataset.summary = description
                            dataset.save()
                        # Retrieve the existing categories of the dataset.
                        existing_categories = dataset.categories.all()
                        for category_obj in dataset_categories:
                            # Add the new category to the existing set.
                            if category_obj not in existing_categories:
                                dataset.categories.add(category_obj)

                    start += 50
                except Exception as e:
                    print(f"Error getting data: {e}")

    @staticmethod
    def create_missing_translations():
        datasets = Datasets.objects.exclude(datasets_english_translations__isnull=False)
        for dataset in tqdm(datasets, desc="Creating translations"):
            try:
                translation = Datasets_English_Translations(
                    dataset=dataset,
                    paper_name=DatasetsScraper.translate_text(dataset.paper_name),
                    summary=DatasetsScraper.translate_text(dataset.summary),
                )
                if translation.paper_name != "" and dataset.summary != "":
                    translation.save()
            except Exception as e:
                print(f"Error translating text: {e}")
                print(dataset.paper_name + " " + dataset.summary)

    @staticmethod
    def translate_text(title):
        origin_aplha2 = detect(title)

        if origin_aplha2 != "en":
            translator = GoogleTranslator()
            try:
                translated_text = translator.translate(title[:500], dest="en").text
            except Exception:
                translator = Translator(from_lang=origin_aplha2, to_lang="en")
                translated_text = translator.translate(title[:500])
                if (
                    "QUERY LENGTH LIMIT EXCEEDED. MAX ALLOWED QUERY : 500 CHARS QUERY LENGTH LIMIT EXCEEDED. MAX ALLOWED QUERY : 500 CHARS"
                    in translated_text
                    or "MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS"
                    in translated_text
                ):
                    translated_text = ""
        else:
            translated_text = title
        return translated_text
