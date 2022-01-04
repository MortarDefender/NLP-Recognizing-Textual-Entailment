import webbrowser as web


def search(fname):
    with open(fname, "r") as f:
        read = f.read()

    for site in read.split("\n"):
        web.open(site)


if __name__ == '__main__':
    search("sites.txt")
