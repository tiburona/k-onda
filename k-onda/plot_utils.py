import re
from datetime import datetime


def formatted_now():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def smart_title_case(s):
    lowercase_words = {'a', 'an', 'the', 'at', 'by', 'for', 'in', 'of', 'on', 'to', 'up', 'and', 'as', 'but', 'or', 'nor', 'is'}
    words = re.split(r'(\W)', s)  # Split string on non-alphanumeric characters, preserving delimiters
    title_words = [word if word.lower() not in lowercase_words or i == 0 or i == len(words) - 1 
                   else word.lower() 
                   for i, word in enumerate(words)]
    title = ''.join(title_words)
    return re.sub(r"[A-Za-z]+('[A-Za-z]+)?",
                  lambda mo: mo.group(0)[0].upper() + mo.group(0)[1:].lower() if not mo.group(0).isupper() else mo.group(0),
                  title)


def ac_str(s):
    for (old, new) in [('pd', 'Pandas'), ('np', 'NumPy'), ('ml', 'Matlab')]:
        s = s.replace(old, new)
