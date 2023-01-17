import csv
import json
import os
import queue
import re
import sys
import time
import urllib.request
import warnings
from typing import Iterable, List, Set, Tuple

import bs4
import numpy as np
import utility as util
from paths import bad_link, limiting_domain, limiting_path, starting_url


def check_year(yr: str) -> bool:
    """Checks if year is within the valid interval"""
    yr = yr.split("-")
    y = int(yr[0])
    if y >= 1979:
        return True
    else:
        return False


def generate_links(
    soup: bs4.BeautifulSoup, proper_url: str, limiting_domain: str
) -> Iterable[str]:
    """
    Takes a url as input. Returns the urls of all of the pages linked
    to the initial page in list format. Note that for the final version
    that does not only crawl the web, we will also want to get information
    off of these web pages.

    Parameters
    ----------
    soup : BeautifulSoup object
        BeautifulSoup representation of web page
    proper_url: str
        URL from which to gather links
    limiting_domain: str
        Limiting domain (links outside of this domain can't be scraped)

    Returns
    -------
    List of links

    """
    # reach out to web page

    links_list = soup.find_all("a")

    # find links
    rv = []
    s = set()
    for link in links_list:
        url = link.get("href")
        new_url = util.remove_fragment(url)

        converted_url = util.convert_if_relative_url(proper_url, new_url)

        converted_url = str(converted_url)
        if converted_url != None:
            if util.is_url_ok_to_follow(converted_url, limiting_domain):
                if converted_url not in s:
                    s.add(converted_url)
                    rv.append(converted_url)
    return rv


def list_to_queue(list_: List) -> queue.Queue:
    """Put all of the starting links into the initial queue"""
    q = queue.Queue()
    for term in list_:
        q.put(term)
    return q


def get_next_link(set_ : Set, queue: queue.Queue) -> Tuple:
    """
    Check if the link has already been in the queue.

    Parameters
    ----------
    set_ : Set
        Visited links
    queue : Queue
        Links to visit

    Returns
    -------
    Tuple
        queue is the updated queue, next_in_queue is the
        next link(if it exists), and a boolean indicating whether or not
        a next link can be returned


    """
    if queue.empty():
        return queue, None, False
    else:

        next_in_queue = queue.get()
        while next_in_queue in set_:
            if not queue.empty():
                next_in_queue = queue.get()
            else:
                return queue, None, False
    return queue, next_in_queue, True


def crawl(num_pages_to_crawl: int,
          starting_url: str,
          limiting_domain: str,
          yr: int) -> List:
    """
    Crawls the website for NBA player data

    Parameters
    ----------
    num_pages_to_crawl :  int
        Maximum number of pages to crawl
    starting_url : str
        URL to start crawling
    limiting_domain : str
        Limiting domain for web crawling (crawler won't go outside of this domain)
    yr : int
        Year of data to scrape

    Returns
    -------
    List of player data from that year

    """

    steps = 0

    proper_url, soup = make_soup(starting_url, limiting_domain, start=True)
    starting_links = generate_links(soup, proper_url, limiting_domain)


    print("found start links")

    visited = {starting_url}
    q = list_to_queue(starting_links)

    new_queue, next_link, indicator = get_next_link(visited, q)
    q = new_queue

    mydata = []

    while (steps <= num_pages_to_crawl and indicator):
        visited.add(next_link)
        new_proper_url, rv = make_soup(next_link, limiting_domain)
        if rv != []:
            _= generate_links(rv, new_proper_url, limiting_domain)
            player_info = soup_to_array(rv, yr)
            if player_info:
                mydata += player_info

        steps += 1
        new_queue, next_link, indicator = get_next_link(visited, q)

        q = new_queue

    print("steps: ", steps)
    return mydata




