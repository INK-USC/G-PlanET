"""
from
https://gist.github.com/kracwarlock/c979b10433fe4ac9fb97
"""


from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from rule_based_extractor import rule_based_extractor
KAS=rule_based_extractor()
tokenizer = PTBTokenizer()

import jsonlines
import json
from tqdm import tqdm
import os

def getId(x) -> str:
    return x['id']

class COCOEvalCap:
    def __init__(self, images, gts, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.params = {'image_id': images}
        self.gts = gts
        self.res = res

    def evaluate(self):
        imgIds = self.params['image_id']
        gts = self.gts
        res = self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')

        gts = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            # (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),
            # (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
            (KAS,'KAS'),
            # (WMD(),   "WMD"),
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        for scorer, method in scorers:
            # print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, imgIds, m)
                    # print("%s: %0.3f" % (m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, imgIds, method)
                # print("%s: %0.3f" % (method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

def calculate_metrics(rng, truths, predicts):
    imgIds = rng
    gts = {}
    res = {}

    imgToAnnsGTS = {ann['image_id']: [] for ann in truths}
    for ann in truths:
        imgToAnnsGTS[ann['image_id']] += [ann]

    imgToAnnsRES = {ann['image_id']: [] for ann in predicts}
    for ann in predicts:
        imgToAnnsRES[ann['image_id']] += [ann]

    for imgId in imgIds:
        gts[imgId] = imgToAnnsGTS[imgId]
        res[imgId] = imgToAnnsRES[imgId]
    evalObj = COCOEvalCap(imgIds, gts, res)
    evalObj.evaluate()
    return evalObj.eval

def steps2list(steps:str)->list:
    steps=steps.split('|')
    ans=[]
    for i in steps:
        if i.startswith('END'):break
        try:
            ans.append(i.split(':')[2].strip())
        except:
            try:
                ans.append(i.split(':')[1].strip())
            except:ans.append(i.replace('STEP','').strip())
    return ans



if __name__ == '__main__':

    L=[]
    for root, dirs, files in os.walk("../result"):
        for file in files:
            if file.endswith('.jsonl') and 'base' not in file:
                L.append(file)

    L=[
    ]
    print(L)
    headers=['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'SPICE','KAS']
    with open(OUTPUT_PATH,'a') as fout:
        for header in headers :
            fout.write(f',{header}')
        fout.write('\n')
        for l in L:
            print(l)
            fileResult={i:0 for i in headers}
            truth={}
            predictions={}
            datasetGTS=[]
            datasetRES=[]
            with open(f'./result/{l}', "r+", encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    truth[getId(item)]=[item['truth']]
                    predictions[getId(item)]=[item['predict']]
            count=0
            for id in tqdm(truth.keys()):
                gt=truth[id]
                for i in range(len(gt)):
                    gt[i]=steps2list(gt[i])
                for pred in predictions[id]:
                    try:
                        preds=steps2list(pred)
                        while len(preds)<len(gt[0]):
                            preds.append(preds[-1])
                        ids=range(len(gt[0]))
                        res= [{'image_id':i,'caption':preds[i]} for i in ids]
                        gts= [{'image_id':i,'caption':g[i]} for g in gt for i in ids]
                        result=calculate_metrics(ids, gts, res)
                        for key in headers:
                            try:
                                fileResult[key]=fileResult[key]+result[key]
                            except:pass
                        count+=1
                    except:continue
            for key in headers:
                fileResult[key]=(fileResult[key]+1e-15)/(count+1e-8)
            fout.write(f'{l},')
            for key in headers:
                try:
                    fout.write(f"{fileResult[key]},")
                except:
                    fout.write(f",")
            fout.write('\n')
            fout.flush()


