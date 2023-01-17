import queue
import re
import warnings
from typing import Iterable, List, Set, Tuple

import bs4
import numpy as np
import utility as util
from paths import bad_link, limiting_domain, limiting_path, starting_url
from tqdm import tqdm

warnings.filterwarnings("ignore")


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


def get_next_link(set_: Set, queue: queue.Queue) -> Tuple:
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


def crawl(
    num_pages_to_crawl: int, starting_url: str, limiting_domain: str, yr: int
) -> List:
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

    # import ipdb; ipdb.set_trace()
    proper_url, soup = make_soup(starting_url, limiting_domain, start=True)
    starting_links = generate_links(soup, proper_url, limiting_domain)

    print("found start links")

    visited = {starting_url}
    q = list_to_queue(starting_links)

    new_queue, next_link, indicator = get_next_link(visited, q)
    q = new_queue

    mydata = []

    while steps <= num_pages_to_crawl and indicator:
        visited.add(next_link)
        new_proper_url, rv = make_soup(next_link, limiting_domain)
        if rv != []:
            _ = generate_links(rv, new_proper_url, limiting_domain)
            player_info = soup_to_array(rv, yr)
            if player_info:
                # name = player_info[0][1]
                # print(name)
                mydata += player_info

        steps += 1
        new_queue, next_link, indicator = get_next_link(visited, q)

        q = new_queue

    print("steps: ", steps)
    return mydata


def make_soup(initial_url: str, limiting_domain: str, start: bool = False) -> Tuple:
    """Get an individual Player's career data for a given url and limiting domain"""

    req1 = util.get_request(initial_url, False)
    if req1 == None:
        print("bad")

    proper_url = util.get_request_url(req1)
    rv = (None, [])

    if util.is_url_ok_to_follow(proper_url, limiting_domain):
        if "Summary" in proper_url:
            text = util.read_request(req1)
            soup = bs4.BeautifulSoup(text, "html5lib")
            rv = (proper_url, soup)
    if start:
        text = util.read_request(req1)
        soup = bs4.BeautifulSoup(text, "html5lib")
        rv = (proper_url, soup)

    return rv


def soup_to_array(soup: bs4.BeautifulSoup, yr: int) -> List:
    """
    Turns a beautiful soup object into a vector of a player's career data
    """
    s = soup.find_all("title")[0].text.split(",")[0]
    name = " ".join(s.split(" ")[:-2])
    tags = soup.find_all("tr", class_="per_game")

    data = {}
    for tag in tags:
        if tag.find_all(
            "td",
            {"id": re.compile(r"teamLinenba_reg_[Per_GameMisc_StatsAdvanced_Stats]+")},
        ):
            t = tag.text.split("\n")[1:-1]
            season = int(t[0].split("-")[0]) + 1
            tally_count = 0
            if season > 1979 and season == yr:
                info = []
                for x in t[2:]:
                    if x != "-":
                        info.append(float(x))
                    else:
                        tally_count += 1
                        info.append(0)
                if season in data:
                    data[season] += info
                else:
                    data[season] = info

    if data != {}:
        array_data = []
        for k, v in data.items():
            d = [k] + [name] + v
            array_data.append(d)

            if tally_count < 5:
                return array_data
            else:
                return None
    else:
        array_data = None


def season_to_arrays(season: Iterable, save_name: str):
    """Saves a season's worth of player data"""
    names = []
    numeric = []
    for player in season:
        name = player[1]
        vect = [player[0]] + player[2:]
        names.append(name)
        numeric.append(vect)

    names = np.array(names)
    numeric = np.array(numeric)

    np.save(save_name + "_names", names)
    np.save(save_name + "_numeric", numeric)


if __name__ == "__main__":
    starting_url = "https://basketball.realgm.com/nba/players/"
    limiting_path = "/player/"

    start_yr = 1987
    end_yr = 2023

    all_data = []
    for yr in tqdm(range(start_yr, end_yr)):
        try:
            print(yr)
            sub_start = starting_url + str(yr)
            players_yr = crawl(1000000, sub_start, limiting_domain, yr)
            season_to_arrays(players_yr, str(yr))
            del players_yr
        except:
            pass
