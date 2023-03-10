import os
import urllib.parse

import bs4
import requests
from paths import limiting_path


def get_request(url, verify_not=True):
    """
    Open a connection to the specified URL and if successful
    read the data.
    Inputs:
        url: must be an absolute URL

    Outputs:
        request object or None
    Examples:
        get_request("http://www.cs.uchicago.edu")
    """

    if is_absolute_url(url):
        try:
            r = requests.get(url, verify=verify_not)
            if r.status_code == 404 or r.status_code == 403:
                r = None
        except:
            # fail on any kind of error
            r = None
    else:
        r = None

    return r


def read_request(request):
    """
    Return data from request object.  Returns result or "" if the read
    fails..
    """

    try:
        return request.text.encode("iso-8859-1")
    except:
        print("read failed: " + request.url)
        return ""


def get_request_url(request):
    """
    Extract true URL from the request
    """
    return request.url


def is_absolute_url(url):
    """
    Is url an absolute URL?
    """
    if url == "":
        return False
    return urllib.parse.urlparse(url).netloc != ""


def remove_fragment(url):
    """remove the fragment from a url"""

    (url, frag) = urllib.parse.urldefrag(url)
    return url


def convert_if_relative_url(current_url, new_url):
    """
    Attempt to determine whether new_url is a relative URL and if so,
    use current_url to determine the path and create a new absolute
    URL.  Will add the protocol, if that is all that is missing.
    Inputs:
        current_url: absolute URL
        new_url:
    Outputs:
        new absolute URL or None, if cannot determine that
        new_url is a relative URL.
    Examples:
        convert_if_relative_url("http://cs.uchicago.edu", "pa/pa1.html") yields
            'http://cs.uchicago.edu/pa/pa.html'
        convert_if_relative_url("http://cs.uchicago.edu", "foo.edu/pa.html") yields
            'http://foo.edu/pa.html'
    """
    if new_url == "" or not is_absolute_url(current_url):
        return None

    if is_absolute_url(new_url):
        return new_url

    parsed_url = urllib.parse.urlparse(new_url)
    path_parts = parsed_url.path.split("/")

    if len(path_parts) == 0:
        return None

    ext = path_parts[0][-4:]
    if ext in [".edu", ".org", ".com", ".net"]:
        return "http://" + new_url
    elif new_url[:3] == "www":
        return "http://" + new_url
    else:
        return urllib.parse.urljoin(current_url, new_url)


def is_url_ok_to_follow(url, limiting_domain):
    """
    Inputs:
        url: absolute URL
        limiting domain: domain name
    Outputs:
        Returns True if the protocol for the URL is HTTP, the domain
        is in the limiting domain, and the path is either a directory
        or a file that has no extension or ends in .html. URLs
        that include an "@" are not OK to follow.
    Examples:
        is_url_ok_to_follow("http://cs.uchicago.edu/pa/pa1", "cs.uchicago.edu") yields
            True
        is_url_ok_to_follow("http://cs.cornell.edu/pa/pa1", "cs.uchicago.edu") yields
            False
    """

    if "mailto:" in url:
        return False

    if "@" in url:
        return False

    # if url[:LEN_ARCHIVES] == ARCHIVES:
    # return False

    parsed_url = urllib.parse.urlparse(url)

    if parsed_url.scheme != "http" and parsed_url.scheme != "https":
        # print("oh no")
        return False

    if parsed_url.netloc == "":
        # print("empty string")
        return False

    # just for nba stuff
    if "Summary" not in url:
        return False

    if parsed_url.fragment != "":
        # print("A")
        return False

    if parsed_url.query != "":
        # print("B")
        return False

    path = parsed_url.path
    if path[:8] != limiting_path:
        # print("D")
        # use regular expressions to just get teams, players?
        # probably a good idea
        return False

    loc = parsed_url.netloc
    ld = len(limiting_domain)
    trunc_loc = loc[-(ld + 1) :]

    if not (limiting_domain == loc or (trunc_loc == "." + limiting_domain)):
        # print("C")
        return False

    # does it have the right extension
    (filename, ext) = os.path.splitext(parsed_url.path)
    return ext == "" or ext == ".html"


def is_subsequence(tag):
    """
    Does the tag represent a subsequence?
    """
    return (
        isinstance(tag, bs4.element.Tag)
        and "class" in tag.attrs
        and tag["class"] == ["courseblock", "subsequence"]
    )


def is_whitespace(tag):
    """
    Does the tag represent whitespace?
    """
    return isinstance(tag, bs4.element.NavigableString) and (tag.strip() == "")


def find_sequence(tag):
    """
    If tag is the header for a sequence, then
    find the tags for the courses in the sequence.
    """
    rv = []
    sib_tag = tag.next_sibling
    while is_subsequence(sib_tag) or is_whitespace(tag):
        if not is_whitespace(tag):
            rv.append(sib_tag)
        sib_tag = sib_tag.next_sibling
    return rv
