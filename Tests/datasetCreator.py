import json
# import pandas as pd


def readFile(fileName):
    if ".jsonl" in fileName:
        with open(fileName) as f:
            data = [json.loads(line) for line in f]
    
    return data

def createLine(label, premise, hypothesis):
    return {"label": label, "premise" : premise, "hypothesis": hypothesis}

def getReleventData(data, fileName, language = "en"):
    dataset = []
    keys = ["gold_label", "sentence1", "sentence2"]
    
    for record in data:
        if "XNLI" in fileName and record["language"] != language:
            continue
        else:
            line = createLine(record[keys[0]], record[keys[1]], record[keys[2]])
            dataset.append(line)
    
    return dataset

def writeDataset(data, fileName):
    json.dump(data, open(fileName, "w"))

def removeAndCreate(files, verbose = False):
    for fileName in files:
        if verbose:
            print(f"start reading {fileName}")
        
        dataset = getReleventData(readFile(fileName), fileName)
        
        if verbose:
            print(f"start writing {fileName}")
        
        writeDataset(dataset, "res//{}".format(fileName.split("//")[1][:-1]))

def combineAllFiles(files, outputFile, verbose = False):
    dataset = []
    
    for fileName in files:
        if verbose:
            print(f"start reading {fileName}")
        
        data = json.load(open(fileName, "r"))
        
        dataset.extend(data)
    
    writeDataset(dataset, outputFile)
    

def main(action = None):
    oldFiles = ["snli//snli_dev.jsonl",  "snli//snli_test.jsonl", "snli//snli_train.jsonl", 
              "XNLI-1.0//xnli.dev.jsonl",  "XNLI-1.0//xnli.test.jsonl",
              "multinli_1.0//multinli_1.0_dev_matched.jsonl",  "multinli_1.0//multinli_1.0_dev_mismatched.jsonl", 
              "multinli_1.0//multinli_1.0_train.jsonl"]
    
    newFiles = ["res//snli_dev.json",  "res//snli_test.json", "res//snli_train.json", 
             "res//xnli.dev.json",  "res//xnli.test.json",
             "res//multinli_1.0_dev_matched.json",  "res//multinli_1.0_dev_mismatched.json", 
             "res//multinli_1.0_train.json"]
    
    if action == "create":
        removeAndCreate(oldFiles)
    elif action == "combine":
        combineAllFiles(newFiles, "res//unifiedDataSet.json")


if __name__ == '__main__':
    main()
