import re
import torch
import pandas as pd
import numpy as np
import pickle

def replace_string_newline(str_start: str, str_end: str, text: str) -> str: 
    """
    re.sub function stops at newline characters, but this function moves past these
    params: str_start, the start character of the string to be delted
            str_end, the end character of the string to be deleted
            text, the string to be edited
    return: text, a string without the string between str_start and str_end
    """
    while text.find(str_start) != -1:
        begin = text.find(str_start) # also just starts w/o http, has (/ 
        nextp = text.find(str_end, begin, len(text))
        text = text.replace(text[begin-1:nextp+1], ' ')
    return text

def filter_news(txt_file: list) -> list:
    """
    filters non-content (e.g. HTTP references, bullets, hashtags) from text data
    params: txt_file, a list of strings, e.g. from news corpora with non-content data
    return: out_txt, a list of strings filtered as designated 
    """
    
    mid_txt = []
    out_txt = []
    for line in txt_file:
        mid_txt.extend(line.split('\n\n'))
    for line in mid_txt:
        while line.find('(http') != -1:
            begin = line.find('(http') # also just starts w/o http, has (/ 
            nextp1 = line.find(')\n', begin, len(line))
            nextp2 = line.find(') ', begin, len(line))
            nextp3 = line.find(')', begin, len(line))
            nextp4 = line.find(' ', begin, len(line))
            nextp = max(nextp1, nextp2, nextp3, nextp4)
            line = line.replace(line[begin-1:nextp+1], ' ')
        line = replace_string_newline('[http', ']', line)
        line = replace_string_newline('<http', '>', line)
        line = replace_string_newline('/http', ')', line)
        line = replace_string_newline('(/', ')', line)
        line = replace_string_newline('(\\', ')', line)
        line = replace_string_newline('(data:', ')', line)
        line = replace_string_newline('(java:', ')', line)
        line = re.sub('-\\n', '-', line) #replaces hyphenated newline with just hyphen
        line = re.sub('\n',' ',line) # replaces newline characters with a space
        line = re.sub('\\xa0', ' ', line) # replaces coded space with regular space
        line = re.sub('\\xad', '', line) # replaces soft-hyphens
        line = re.sub('^ *', '', line) # removes leading space
        line = re.sub('  *', ' ', line) # removes extra space
        line = re.sub('^ *', '', line) # removes leading space
        line = re.sub('\*(.*?)\\n', '', line) # removes phrases beginning with asterisk
        line = re.sub('^\[(.*?)\] .*$', '', line) # removes lines beginning and ending with brackets
        line = re.sub('^[#*](.*?)$', '', line) # removes lines beginning with hashtag
        line = re.sub('\[','', line) # delete extra brackets does not remove lines beginning with open bracket
        line = re.sub('[*#!>_\]\|]*', '', line) # removes extra characters
        line = re.sub('(.*?)Video duration(.*?)$', '', line) # removes lines of video captions
        line = re.sub(' \.', '.', line) # removes extra space before periods
        line = re.sub(' ,', ',', line) # removes extra space before commas
        line = re.sub('^[ >]$', '', line) # removes entries with single space or single >
        line = re.sub('  *', ' ', line) # removes extra space
        line = re.sub('^ *', '', line) # removes leading space
        line = re.sub(' *$', '', line) # removes ending space
        line = re.sub('((.*?).html)', '', line) # removes other link data
         # remove single words from a list???? how??? Image captions? Image numbers?
        out_txt.append(line)
    return out_txt

file1 = pd.read_csv('dict_adj_to_country.csv')
adj_to_country = dict(zip(file1.Adjective, file1.Country))