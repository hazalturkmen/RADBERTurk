import glob
import re
from tqdm import tqdm


def preproc(sentence):
    """

    :param sentence:
    :return:
    """
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = re.sub("[0-9]", "#", sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence


def preproc_files():
    files = glob.glob("/home/hazal/nlp_dataset/brain_CT/splitted_txt_2/*.txt")
    print("\nPre-processing medical text...")
    for f in tqdm(files):
        f_txt = open(f, "r")
        file_contents = f_txt.read()
        contents_split = file_contents.splitlines()
        for sent in contents_split:
            preproc(sent)


if __name__ == '__main__':
    preproc_files()
