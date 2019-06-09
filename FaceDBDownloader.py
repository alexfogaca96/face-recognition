import os
import urllib.request

# Matches
with open('matches.txt', 'r') as matches_file:
    matches_path = "matches"
    if not os.path.isdir(matches_path):
        os.mkdir(matches_path)

    for line in matches_file:
        links = line.split(" ")
        line_matches_path = matches_path + '/' + links[0][:-1]
        print("downloading... " + line_matches_path)
        if not os.path.isdir(line_matches_path):
            os.mkdir(line_matches_path)

        match_one = open(line_matches_path + '/match_one.jpg', 'wb')
        match_one.write(urllib.request.urlopen(links[1]).read())
        match_one.close()
        match_two = open(line_matches_path + '/match_two.jpg', 'wb')
        match_two.write(urllib.request.urlopen(links[2]).read())
        match_two.close()
        print(" downloaded    " + line_matches_path)
matches_file.close()

# Mismatches
with open('mismatches.txt', 'r') as mismatches_file:
    mismatches_path = "mismatches"
    if not os.path.isdir(mismatches_path):
        os.mkdir(mismatches_path)

    for line in mismatches_file:
        links = line.split(" ")
        line_mismatch_path = mismatches_path + '/' + links[0][:-1]
        print("downloading... " + line_mismatch_path)
        if not os.path.isdir(line_mismatch_path):
            os.mkdir(line_mismatch_path)

        mismatch_one = open(line_mismatch_path + '/mismatch_one.jpg', 'wb')
        mismatch_one.write(urllib.request.urlopen(links[1]).read())
        mismatch_one.close()
        mismatch_two = open(line_mismatch_path + '/mismatch_two.jpg', 'wb')
        mismatch_two.write(urllib.request.urlopen(links[2]).read())
        mismatch_two.close()
        print(" downloaded    " + line_mismatch_path)
mismatches_file.close()
