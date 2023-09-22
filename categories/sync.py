from .authorities_sync.unesco import UnescoScraper
from .authorities_sync.eric import EricScraper
from .authorities_sync.oecd import OecdScraper
from .models import Authorities


def scrap_unesco():
    """
    Starts the scraping process
    """
    scraper = UnescoScraper()
    scraper.scrape()


def scrap_eric():
    """
    Starts the scraping process
    """
    scraper = EricScraper()
    scraper.scrape()


def scrap_oecd():
    """
    Starts the scraping process
    """
    scraper = OecdScraper()
    scraper.scrape()


def start_scraping():
    if Authorities.objects.filter(name="UNESCO", auto_sync=True).count():
        scrap_unesco()
    if Authorities.objects.filter(name="ERIC", auto_sync=True).count():
        scrap_eric()
    if Authorities.objects.filter(name="OECD", auto_sync=True).count():
        scrap_oecd()
