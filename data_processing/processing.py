import re
import os
import configparser
import warnings
import string
from urllib.parse import urlparse
from bs4 import BeautifulSoup,MarkupResemblesLocatorWarning
from unicodedata import normalize
import pandas as pd
import numpy as np
from profanity import profanity

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

def read_config(section="params", config_path='./../'):
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    if section not in config.sections():
        print(f"Section '{section}' not found in configuration.")
        raise KeyError(f"Section not found: {section}")
    return {key: config[section][key] for key in config[section]}

url_regex = re.compile(r'http\S+|www.\S+')
whitespace_regex = re.compile(r'\s+')
punctuation_regex = re.compile(f"([{string.punctuation}])")
non_alphanumeric_regex = re.compile(r'[^a-zA-Z0-9.,!?\' ]')
punctuation_regex = re.compile(f"([{string.punctuation}])")
contraction_mapping = pd.read_json('./data_processing/contraction_mapping.json', typ='series').to_dict()
config_path='./config.ini'
config_params = read_config(section="sampling", config_path=config_path)
   
wiki_markup_regex = re.compile(
    r'thumb\|[\dpx]*\|?|'
    r'right\|[\dpx]*\|?|'
    r'left\|[\dpx]*\|?|'
    r'center\|[\dpx]*\|?|'
    r'[\dpx]+\|'
)

def prepare_data(data, input_col, clean_col):
    data[input_col] = data[input_col].astype(str)
    data[clean_col] = data[input_col].apply(normalize_text).apply(remove_whitespace).apply(pad_punctuation)
    return data[data[clean_col].notnull() & data[clean_col].str.strip().astype(bool)]

def string_to_bool(string_value):
    return string_value.lower() in ['true', '1', 't', 'y', 'yes', 'on']

def pad_punctuation(s):
    if string_to_bool(config_params.get("padding", "False")):
        if not isinstance(s, str):
            return ""
        s = punctuation_regex.sub(r" \1 ", s)
        print(s)
        return whitespace_regex.sub(' ', s).strip()
    return s
    
def normalize_text(text):
    if isinstance(text, str):
        try:
            # Existing normalization steps
            text = url_regex.sub(lambda m: urlparse(m.group(0)).netloc.replace('www.', ''), text)
            text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            text = wiki_markup_regex.sub('', text)  # Remove wiki markup
            text = re.sub(r'\n\n.*?\n\n*?\n', ' ', text)
            text = text.replace('\n', ' ')
            text = ' '.join(BeautifulSoup(text, 'html.parser').stripped_strings)
            text = re.sub(r'>>\d+', ' ', text)
            # Revised pattern to remove 'thumb|', 'px', '200px|', 'right|', and similar patterns
            text = re.sub(r'thumb\|\d*x\d*px\|right\|', '', text)
            text = re.sub(r'thumb\|\d*x\d*px\|', '', text)
            text = re.sub(r'thumb\|', '', text)
            text = re.sub(r'\d*x\d*px\|', '', text)
            text = re.sub(r'^\s*>+', '', text, flags=re.MULTILINE)
            # Existing normalization steps continued
            if string_to_bool(config_params.get("contraction_mapping", "False")):
                text = ' '.join(contraction_mapping.get(t, t) for t in text.split())
            if string_to_bool(config_params.get("non_alpha_numeric", "False")):
                text = non_alphanumeric_regex.sub(' ', text)
            return whitespace_regex.sub(' ', text).strip()
        except ValueError:
            return text
    return text

def remove_whitespace(text):
    if isinstance(text, str):
        return " ".join(text.split())
    return text

def remove_profanity(text):
  words = text.split()
  cleaned_words = [("*" * len(word)) if profanity.contains_profanity(word) else word for word in words]
  return " ".join(cleaned_words)

def is_valid_chunk(text: str) -> bool:
    """Check if a chunk has enough meaningful content."""
    # Remove XML tags and whitespace
    content = text.split("<content>")[-1].split("</content>")[0].strip()
    
    # Minimum requirements
    min_words = 5
    min_chars = 20
    
    # Count actual words (excluding common noise)
    words = [w for w in content.split() if len(w) > 2]  # Filter out very short words
    
    return len(words) >= min_words and len(content) >= min_chars

def clean_chunk_text(text: str) -> str:
    """Clean and format chunk text to remove excessive whitespace."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.isspace():
            if any(tag in line for tag in ['<temporal_context>', '</temporal_context>', '<content>', '</content>']):
                cleaned_lines.append(line)
            else:
                cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)