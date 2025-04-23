import re
import unicodedata
from collections import Counter

file_path = "C:/dataset.txt"  # Update this with your file path

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Length of dataset:", len(text), "\n")
print("First 100 characters:\n", text[:100])

chars = sorted(list(set(text)))  
vocab_size = len(chars)          
print(f"\nUnique characters:\n{''.join(chars)}")
print(f"\nVocabulary size: {vocab_size}")

print(f"\nFile have {len(unique_words)} unique words.")

decomposed_chars = {char: unicodedata.decomposition(char) for char in text if unicodedata.decomposition(char)}
print(f"\nDecomposed characters found:\n{decomposed_chars}")

non_standard_chars = {char for char in text if not char.isprintable() and char not in '\n\t '}
print(f"\nNon-standard characters:\n{non_standard_chars}")

char_frequency = Counter(text)  # Count character frequencies
sorted_char_frequency = dict(sorted(char_frequency.items(), key=lambda item: ord(item[0])))  # Sort by Unicode

print(f"\n{'Character':<10}{'Unicode':<10}{'Frequency':<10}")
print("-" * 30)
for char, freq in sorted_char_frequency.items():
    unicode_code = f"U+{ord(char):04X}"  # Unicode code of the character
    printable_char = repr(char) if char.isprintable() else f"\\u{ord(char):04X}"  # Printable version
    print(f"{printable_char:<10}{unicode_code:<10}{freq:<10}")