import glob

from tqdm import tqdm

print("\nPrinting CT labeling model...")
files = glob.glob("/home/hazal/nlp_dataset/brain_CT/example/experiment/*.txt")
fon = open('/home/hazal/nlp_dataset/brain_CT/example/experiment/result.txt', 'w')

for f in tqdm(files):
    with open(f, "r") as fo:
        word = "Accuracy:"
        words = fo.readlines()
        for line in words:
            if word in line:
                print(line[9:15])
                fon.write(line[9:15])
                fon.write("\n")
fon.close()





