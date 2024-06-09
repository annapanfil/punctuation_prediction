import argparse
import numpy as np
import os
import re

PUNCT_MAPPING = {'?': '?',
                 '.': '.',
                 ',': ',',
 
                 '!': '!',
                 '…': '…',
                 '-': '-',
                 ':': ":",
                 } # default is a dot

PUNCT_NAMES = {'.': "PERIOD",
               ',': "COMMA",
               '?': "QUESTION",
               "!": "EXCLAMATION",
               "-": "DASH",
               ":": "COLON",
               "…": "ELLIPSIS"}


def split_words_and_punctuation(text: str) -> tuple:
    # @return elements – list of words and punctuation signs
    # @return punctuation_mask – list of 0s and 1s, where 0 indicates word and 1 indicates punctuation sign

    excluded_signs = '\%\+\"\*\/\\=$€\(\)\[\]\'@·;&'
    word_signs = f"[\w{excluded_signs}]+"
    punctuation_signs = f"(?![\s{excluded_signs}])\W|…"
    elements = re.compile(rf"{punctuation_signs}|{word_signs}|\n").findall(text) # alfanumeric or non-alphanumeric characters (punctuation)
    elements = [element for element in elements if element.strip(" ")] # discard empty strings

    punctuation_mask = []
    for i, element in enumerate(elements):
        if re.match(rf"{word_signs}|\n", element):
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


def save_to_file(out_fname: str, elements: list, punctuation_mask: list, punct_names: dict, newlines: bool):
    ### save in format of the alam-etal-2020-punctuation code
    ### i.e. word in lowercase, tab, punctuation sign name (e.g. "PERIOD" or "O")
    # @param out_fname – path of the output file
    # @param elements – list of words and punctuation signs
    # @param punctuation_mask – list of 0s and 1s, where 0 indicates word and 1 indicates punctuation sign
    # @param punct_names – dictionary with punctuation signs and their names

    with open(out_fname, 'w') as f:
        nlines = 0
        for i in range(len(elements)):
            if punctuation_mask[i] == 1:
                continue
            elif elements[i] == "\n":
                f.write("\n")
                nlines += 1
                continue

            # write a word and its punctuation sign
            f.write(elements[i].lower() + '\t')
            if i+1 < len(elements) and punctuation_mask[i+1] == 1:
                f.write(punct_names[elements[i+1]] + "\n")
            else:
                f.write("O\n")
            nlines += 1
        
        if newlines:
            f.write("\n")
            nlines += 1

    print("Saved to " + out_fname)

    return nlines

def merge_files(filenames, out_filename):
    if os.path.exists(out_filename):
        print(f"removing {out_filename}")
        os.remove(out_filename)

    for filename in filenames:
        with open(filename, 'r') as f:
            content = f.read()
        with open(out_filename, 'a') as f:
            f.write(content)
    print(f"merged {filenames} into {out_filename}")
        

def get_lines_poleval2022(split, verb=False, prefix=""):
    with open(prefix + "data/raw/poleval2022/" + split + "/in.tsv", 'r') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]
        lines = [line.split("\t") for line in lines]

        sources = list(zip(*lines))[0]
        texts_ts = list(zip(*lines))[2]

        texts = [re.sub(r':\d+-\d+', '', text) for text in texts_ts] # remove timestamps
        timestamps = [re.findall(r'\d+-\d+', text) for text in texts_ts]

        if verb:
            print(len(texts), "sentences\n", len(set(sources)), "audio files")

    return texts, texts_ts, timestamps

def get_pause_durations(timestamps, texts, path, limit=None):
    ### get durations of pauses between words from the dataset and align them according to the created dataset
    # @param timestamps – list of timestamps in the format "beginning-end"
    # @param texts – list of texts
    # @param path – path to the created dataset file
    ###

    timestamps = [item for sublist in timestamps for item in sublist]
    durations = []
    beginnings = []
    ends = []
    # word_durations = []

    for timestamp in timestamps:
        numbers = timestamp.split("-")
        beginnings.append(int(numbers[0]))
        ends.append(int(numbers[1]))
        # word_durations.append(int(numbers[1]) - int(numbers[0]))

    for i in range(len(timestamps)-1):
        durations.append(beginnings[i+1] - ends[i])
        if durations[i] < 0: # it means the new recording started
            durations[i] = np.nan        
    durations.append(np.nan) # for the last sign

    words_old = []  # from in dataset
    for text in texts:
        words_old.extend(text.split())

    with open(path, "r") as f: # from created dataset
        lines = f.read().splitlines()

    lines = [line.split("\t") for line in lines]
    words, interpunction = list(zip(*lines))

    list(zip(words, words_old, interpunction, durations))[4636:4637]

    excluded_signs = '\%\+\"\*\/\\=$€\(\)\[\]\@·;'
    word_signs = f"[\w{excluded_signs}]+"
    punctuation_signs = f"(?![\s{excluded_signs}])\W|…"

    # Words from "in" dataset sometimes have punctuation inside (e.g. "…", "à", "Dobraю", "-" inside words).
    # They get split in the "allpunct" dataset. We have split the words in the "in" dataset as well,
    # to match the lengths of the durations and words lists.
    # ... denotes a realy long pause and the duration of the pause is usually in the previous word, that's why i-1 pause is removed.
    
    # remove punctuation signs from words
    i=0
    while i!=len(words):
        if words[i] != words_old[i].lower():
            if words_old[i] == "…":
                words_old.pop(i)
                durations.pop(i-1)
                i-=1
                # print("found …", durations[i-1], durations[i], durations[i+1])

            elif words_old[i] == "à" or words_old[i] == "Dobraю":
                pass

            else:
                elements = re.compile(rf"{punctuation_signs}|{word_signs}|\n").findall(words_old[i]) # alfanumeric or non-alphanumeric characters (punctuation)
                # print(i, words[i], words_old[i], elements)
                words_old.insert(i, elements[2])
                words_old.insert(i, elements[0])
                words_old.pop(i+2)
                durations.insert(i, durations[i])
                durations.insert(i, None)
                durations.pop(i+2)
        i+=1

    if limit is not None:
        # remove durations above limit and replace None with 0
        durations = np.array(durations, dtype=float)
        durations = np.where(durations < limit, durations, limit)
        durations = np.nan_to_num(durations, nan=0)

    return words_old, interpunction, durations

def create_durations_file(original_split, split_name, prefix=""):
    texts, _, timestamps = get_lines_poleval2022(original_split)
    _, _, durations = get_pause_durations(timestamps, texts, f"{prefix}data/pl/{split_name}_2022_allpunct", limit = 5000)

    # Min-max normalization
    min_val = durations.min()
    max_val = durations.max()
    durations = (durations - min_val) / (max_val - min_val)

    ndurations = len(durations)
    durations = "\n".join([str(dur) for dur in durations])

    with open(f"{prefix}data/pl/{split_name}_2022_allpunct_durations", "w") as f:
        f.write(durations)
        f.write("\n")

    print("Saved durations to " + f"{prefix}data/pl/{split_name}_2022_allpunct_durations")

    return ndurations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Punctuation restoration test')
    parser.add_argument('--newlines', default=False, type=lambda x: (str(x).lower() == 'true'), help='add new line after each statement in the output file')    
    parser.add_argument('--durations', default=False, type=lambda x: (str(x).lower() == 'true'), help='create file with durations of pauses between words for poleval2022 train and dev-0 sets')    
    parser.add_argument('--secret-tests', default=False, type=lambda x: (str(x).lower() == 'true'), help='create files with secret tests')

    args = parser.parse_args()

    file_paths = ["poleval2021/train", "poleval2021/test-A", "poleval2022/train", "poleval2022/dev-0"]
    if args.secret_tests:
        file_paths += ["poleval2021/test-B", "poleval2021/test-C", "poleval2021/test-D", "poleval2022/test-A", "poleval2022/test-B"] 

    for file_path in file_paths: 
        with open(os.path.join("data", "raw", file_path, "expected.tsv"), 'r', encoding='utf-8') as f:
            lines = [line for line in f.read().split('\n') if line.strip()]
        
        if args.newlines:
            text = "\n".join(lines)
        else:
            text = " ".join(lines)
        text = text.replace("...", "…")

        elements, punctuation_mask = split_words_and_punctuation(text)
        elements = limit_punctuation_signs(elements, punctuation_mask, PUNCT_MAPPING, verb=True)

        os.makedirs("data/pl", exist_ok=True)
        norm_path = os.path.normpath(file_path)
        split_name = "val" if norm_path.split(os.sep)[1] in ("dev-0", "2021/test-A") else norm_path.split(os.sep)[1]

        out_fname = os.path.join("data", "pl", split_name + "_" + norm_path.split(os.sep)[0][-4:]+"_allpunct")
        nlines = save_to_file(out_fname, elements, punctuation_mask, PUNCT_NAMES, args.newlines)

        if args.durations and "2022" in file_path:
            ndurations = create_durations_file(norm_path.split(os.sep)[1], split_name)

            assert nlines == ndurations, f"Number of lines in the dataset ({nlines}) and durations ({ndurations}) do not match."
    
    merge_files(["data/pl/train_2021_allpunct", "data/pl/train_2022_allpunct"], "data/pl/train_allpunct")
    merge_files(["data/pl/val_2021_allpunct", "data/pl/val_2022_allpunct"], "data/pl/val_allpunct")
    if args.secret_tests:
        merge_files(["data/pl/test-B_2021_allpunct", "data/pl/test-C_2021_allpunct", "data/pl/test-D_2021_allpunct", "data/pl/test-A_2022_allpunct", "data/pl/test-B_2022_allpunct"], "data/pl/test_allpunct")
        merge_files(["data/pl/train_allpunct", "data/pl/val_allpunct"], "data/pl/trainval_allpunct")