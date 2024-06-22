from spellchecker import SpellChecker

def perform_spell_check(text):
    spell = SpellChecker()

    # Split text into words
    words = text.split()

    # Find misspelled words
    misspelled = spell.unknown(words)

    # Correct misspelled words
    corrected_text = []
    for word in words:
        if word in misspelled:
            # Get the most likely correction
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word)
        else:
            corrected_text.append(word)

    # Join corrected words back into a string
    corrected_text = ' '.join(corrected_text)
    return corrected_text

# Example usage
input_text = "I havv an apple and banana"
corrected_text = perform_spell_check(input_text)
print("Original Text:", input_text)
print("Corrected Text:", corrected_text)
