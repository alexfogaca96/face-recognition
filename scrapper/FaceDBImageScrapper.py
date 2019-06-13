from bs4 import BeautifulSoup
import requests

baseURL = 'http://vis-www.cs.umass.edu/lfw/'
setsURL = baseURL + 'sets_'
row = 0
for page in range(1, 11):
    URL = setsURL + str(page) + '.html'
    source = requests.get(URL).text
    soup = BeautifulSoup(source, 'lxml')

    table = soup.find('table')
    matches = {}
    mismatches = {}
    for tr in table.find_all('tr', recursive=False):
        match = []
        mismatch = []
        indexTd = 0
        for td in tr.findChildren('td', recursive=False):
            img = td.find('img')
            if img is None:
                indexTd += 1
                continue
            src = img['src']
            if indexTd == 0 or indexTd == 1:
                match.append(baseURL + src)
            elif indexTd == 3 or indexTd == 4:
                mismatch.append(baseURL + src)
            indexTd += 1

        if len(match) == 2 and len(mismatch) == 2:
            matches[row] = match
            mismatches[row] = mismatch
            row += 1

    print('page (' + str(page) + '):')
    print(str(len(matches)) + ' matches')
    print(str(len(mismatches)) + ' mismatches')

    matchesFile = open('matches.txt', 'a')
    for key, value in matches.items():
        matchesFile.write(str(key) + ': ' + value[0] + " " + value[1] + "\n")
    matchesFile.close()

    mismatchesFile = open('mismatches.txt', 'a')
    for key, value in mismatches.items():
        mismatchesFile.write(str(key) + ': ' + value[0] + " " + value[1] + "\n")
    mismatchesFile.close()
