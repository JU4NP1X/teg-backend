import requests
from requests.exceptions import Timeout
from thesaurus.models import Thesaurus
from .models import Datasets


class DatasetsScraper:
    def __init__(self, query):
        self.base_url = "https://onesearch.library.uwa.edu.au/primaws/rest/pub/pnxs"
        self.query = query
        self.timeout = 15
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
            }
        )

    def scrape(self):
        start = 0
        while start < 50:
            try:
                params = {
                    "blendFacetsSeparately": "false",
                    "disableCache": "false",
                    "getMore": "0",
                    "inst": "61UWA_INST",
                    "lang": "en",
                    "limit": "10",
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
                    "vid": "61UWA_INST:UWA",
                }
                response = self.session.get(
                    self.base_url, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()

                if not data["docs"]:
                    break

                for doc in data["docs"]:
                    try:
                        title = doc["pnx"]["display"]["title"][0]
                        description = doc["pnx"]["display"]["description"][0]
                        categories = []
                        for category in doc["pnx"]["display"]["subject"]:
                            categories.extend(category.split(";"))
                    except KeyError:
                        continue

                    # Verificar si el documento tiene al menos un thesaurus asociado a su subject
                    if not categories:
                        continue

                    dataset_categories = []
                    for category in categories:
                        # Filtrar los thesaurus que cumplan con el criterio de búsqueda similar
                        thesaurus = Thesaurus.objects.filter(
                            name__icontains=category
                        ).first()
                        if thesaurus:
                            dataset_categories.append(thesaurus)

                    # Verificar si el dataset tiene al menos un thesaurus asociado antes de crearlo
                    if not dataset_categories:
                        continue
                    dataset, _ = Datasets.objects.update_or_create(
                        paper_name=title, 
                        defaults={'summary': description}
                    )
                    
                    for category_obj in dataset_categories:
                        dataset.categories.add(category_obj)

                start += 10
            except (Timeout, requests.exceptions.RequestException):
                break
