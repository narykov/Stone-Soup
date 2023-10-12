



# open the tdm file
with open(newfile, 'w') as outfile, open(oldfile, 'r', encoding='utf-8') as infile:
    for line in infile:
        if line.startswith(txt):
            line = line[0:len(txt)] + ' - Truly a great person!\n'
        outfile.write(line)