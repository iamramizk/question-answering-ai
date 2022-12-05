import trafilatura
from trafilatura.settings import use_config
import logging

logging.getLogger("trafilatura").setLevel(logging.WARNING)


def text_from_url(url: str):
    """
    Returns main text from a url
    """
    newconfig = use_config()
    newconfig.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded, include_comments=False, config=newconfig)
    return text
