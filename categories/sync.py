from .authorities_sync.unesco import UnescoScraper
from .authorities_sync.eric import EricScraper
from .authorities_sync.oecd import OecdScraper


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
