import re


def get_first_n_pages(filename, n):
    etd_file = open(filename, "r")
    etd_content = etd_file.read()
    pages = re.compile("[0-9]+.json\n").split(etd_content)[1:]

    first_n_pages = pages[:n]

    return first_n_pages
