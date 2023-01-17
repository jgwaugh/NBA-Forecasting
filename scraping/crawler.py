import re
import utility as util
import bs4
import queue
import json
import sys
import csv
import numpy as np
import urllib.request
import os
import time
from typing import Iterable
import warnings

from paths import (
    limiting_domain,
    limiting_path,
    starting_url,
    bad_link
)

def check_year(yr: str) -> bool:
    """Checks if year is within the valid interval """
    yr = yr.split("-")
    y = int(yr[0])
    if y >= 1979:
        return True
    else:
        return False


def generate_links(soup, proper_url, limiting_domain):
    '''
    Takes a url as input. Returns the urls of all of the pages linked
    to the initial page in list format. Note that for the final version
    that does not only crawl the web, we will also want to get information
    off of these web pages.
    '''
    # reach out to web page

    links_list = soup.find_all("a")

    # find links
    rv = []
    s = set()
    for link in links_list:
        url = link.get('href')
        new_url = util.remove_fragment(url)

        converted_url = util.convert_if_relative_url(proper_url, new_url)

        converted_url = str(converted_url)
        if converted_url != None:
            if util.is_url_ok_to_follow(converted_url, limiting_domain):
                if converted_url not in s:
                    s.add(converted_url)
                    rv.append(converted_url)
    return rv