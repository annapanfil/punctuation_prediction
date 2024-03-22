import os
import re


def split_words_and_punctuation(text: str) -> tuple:
    # @return elements – list of words and punctuation signs
    # @return punctuation_mask – list of 0s and 1s, where 0 indicates word and 1 indicates punctuation sign

    excluded_signs = '\%\+\"\-\*\/\\=$€\(\)\[\]\':@·'
    word_signs = f"[\w{excluded_signs}]+"
    punctuation_signs = f"(?![\s\%\+\"\-\*\/\\=$€])\W"
    elements = re.compile(rf"{punctuation_signs}|{word_signs}").findall(text) # alfanumeric or non-alphanumeric characters (punctuation)
    elements = [element for element in elements if element.strip()] # discard empty strings

    punctuation_mask = []
    for i, element in enumerate(elements):
        if re.match(rf"{word_signs}", element):
            punctuation_mask.append(0)
        else:
            elements[i] = elements[i].replace(" ", "")
            punctuation_mask.append(1)

    return elements, punctuation_mask


def limit_punctuation_signs(elements: list, punctuation_mask: list, punct_mapping: dict, verb=False) -> list:
    ### replace rare punctuation signs with these learned by model
    # @return elements – list of words and changed punctuation signs (!but it's also modified in place!)

    for i, (element, mask) in enumerate(zip(elements, punctuation_mask)):
        if mask == 1:
            if element in punct_mapping:
                elements[i] = punct_mapping[element]
            else :
                if verb:
                    print(f"Unknown punctuation sign: {element}")
                elements[i] = '.'

    return elements


def save_to_file(out_fname: str, elements: list, punctuation_mask: list, punct_names: dict):
    ### save in format of the alam-etal-2020-punctuation code
    ### i.e. word in lowercase, tab, punctuation sign name (e.g. "PERIOD" or "O")
    # @param out_fname – path of the output file
    # @param elements – list of words and punctuation signs
    # @param punctuation_mask – list of 0s and 1s, where 0 indicates word and 1 indicates punctuation sign
    # @param punct_names – dictionary with punctuation signs and their names

    with open(out_fname, 'w') as f:
        for i in range(len(elements)):
            if punctuation_mask[i] == 1:
                continue
            
            # write a word and its punctuation sign
            f.write(elements[i].lower() + '\t')
            if i+1 < len(elements) and punctuation_mask[i+1] == 1:
                f.write(punct_names[elements[i+1]] + "\n")
            else:
                f.write("O\n")

    print("Saved to " + out_fname)


PUNCT_MAPPING = {'?': '?',
                 '.': '.',
                 ',': ',',
 
                 '!': '.',
                 '…': '.',
                 '-': ',',
                 ':': ",",
                 ';': '.',
                 } # default is a dot

PUNCT_NAMES = {'.': "PERIOD",
               ',': "COMMA",
               '?': "QUESTION"}

if __name__ == "__main__":
    # train and dev data
    for file_path in ["poleval2021/train", "poleval2021/test-A", "poleval2022/train", "poleval2022/dev-0"]: 
        with open(os.path.join("data", "raw", file_path, "expected.tsv"), 'r', encoding='utf-8') as f:
            lines = [line for line in f.read().split('\n') if line.strip()]
        text = " ".join(lines) #TODO: treat each line as a separate example

        elements, punctuation_mask = split_words_and_punctuation(text)
        elements = limit_punctuation_signs(elements, punctuation_mask, PUNCT_MAPPING, verb=True)

        os.makedirs("data/pl", exist_ok=True)
        norm_path = os.path.normpath(file_path)
        out_fname = os.path.join("data", "pl", norm_path.split(os.sep)[1] + "_" + norm_path.split(os.sep)[0][-4:])
        save_to_file(out_fname, elements, punctuation_mask, PUNCT_NAMES)
    
    
