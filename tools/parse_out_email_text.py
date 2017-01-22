#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
import nltk
import sys

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        if sys.version_info[0] < 3:
            text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
        else:
            text_string=content[1].translate(str.maketrans({key: None for key in string.punctuation}))
    	### space between each stemmed word)
        t=text_string.split()
        stemmer=SnowballStemmer("english")
        t=[stemmer.stem(i) for i in t]
        words=" ".join(t)
    return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print(text)



if __name__ == '__main__':
    main()

