import difflib

word_list = open('/usr/share/dict/words', 'r').read().splitlines()
print(difflib.get_close_matches('bby', word_list))

