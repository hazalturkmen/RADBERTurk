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
    punc = '''!()-[]{};:'"\,<>./?@$%^&±*_~'''
    sentence = re.sub("[0-9]", "#", sentence)
    for ele in sentence:
        if ele in punc:
            sentence = sentence.replace(ele, "")
    return sentence


def preproc_files():
    files = glob.glob("/home/hazal/nlp_dataset/brain_CT/splitted_txt_2/*.txt")
    print("\nPre-processing medical text...")
    for f in tqdm(files):
        f_txt = open(f, "r")
        file_contents = f_txt.read()
        contents_split = file_contents.splitlines()
        for sent in contents_split:
            with open(f.replace('splitted_txt_2', 'splitted_txt_processed'), 'a') as fo:
                fo.write(preproc(sent))
                fo.write("\n")
        fo.close()

if __name__ == '__main__':
    preproc_files()
