import re

from typing import Tuple, Dict, List
from numpy import average


remove={"mightn't", 're', 'they', 'through', 've', 'own', 'll', 'hers', 'those', "doesn't", 'once', 'has', 'too', 'if', 'having', 'from', 'any', 'then', 'couldn', 'mustn', 'himself', 'herself', 'y', 'our', 'so', 'themselves', 'had', 'won', 'me', "hadn't", 'an', 'or', 'will', 'now', 'below', 'to', 'just', 'them', 'over', 'as', 'why', 'both', 'other', 'her', "you've", 'under', 'hasn', "you'll", "needn't", 'wouldn', 'for', 'that', 'how', 'haven', 'was', "she's", 'does', 'o', 'not', 'who', 'is', 'while', "weren't", "should've", 'being', 'same', "that'll", 'than', 'all', 'of', 'my', 'weren', "isn't", 'yours', 'can', "you're", 'ourselves', 'its', 'it', 'whom', 'be', 'no', 'against', 'very', 'few', 'into', 'on', 'after', "wasn't", 'your', 's', 'yourselves', 'what', "wouldn't", 'a', 'until', 'nor', 'do', 'are', 'this', 'i', 'needn', 'most', 'mightn', "shan't", 'their', 'about', "haven't", 'these', "aren't", 'doesn', 'more', 't', "hasn't", 'have', 'didn', 'but', "it's", 'him', 'he', 'should', 'shan', 'itself', "you'd", 'were', 'yourself', 'each', 'out', 'down', 'myself', 'where', 'some', 'hadn', 'because', "mustn't", 'up', "won't", 'm', 'did', 'only', 'she', 'further', 'in', 'and', 'am', 'with', "shouldn't", 'at', 'you', 'doing', 'we', 'his', 'off', 'during', "don't", 'before', 'ma', 'wasn', 'ours', 'when', 'shouldn', 'ain', 'by', 'there', "couldn't", 'd', 'aren', "didn't", 'such', 'between', 'which', 'been', 'the', 'here', 'isn', 'again', 'above', 'theirs', 'don',',', 'green', 'red','black','white','wooden','wood','walk','back','towards'}

directions=['left','right','around']
moves=['turn','go']

def splitword(x):
    ans=[]
    x=x.split(' ')
    stat=0
    tmp=''
    for i in x:
        if i in moves:
            stat=1
            tmp=i
            continue
        if stat==1:
            if i not in directions:
               ans.append(tmp)
               ans.append(i) 
            else:
                ans.append(tmp+' '+i)
            stat=0
            continue
        if i==',' or i in directions:continue
        ans.append(i)
    return ans



def get_chunks(x):
    # print(x)
    x = re.sub(r"step\s[0-9]+[:]\s", "", x.lower())
    x=x.replace(',',' , ').replace('  ',' ').replace('walk','go').replace('.','')
    x =  x.split(' ')
    x = [n for n in x if n not in remove]
    x = ' '.join(x)
    x = splitword(x)
    return x



class rule_based_extractor:
    def __init__(self) -> None:

        pass

    def compute_score(
        self, gts: Dict[int, List[str]], res: Dict[int, List[str]]
    ) -> Tuple[float, List[float]]:

        scores = []
        ground_truths = None
        predicted_sentence = None

        for idx in range(len(gts)):
            ground_truths = gts[idx]
            predicted_sentence = res[idx]

            if predicted_sentence is None:
                continue

            scores.append(self._calculate_percentage_match_list(
                        ground_truths,
                        predicted_sentence[0]
                    ))

        return average(scores), scores


    def _calculate_percentage_match_list(
        self,
        ground_truth_sentences: List[str],
        predicted_sentence: str,
    ) -> float:
        ground_truth_ngrams=[]
        for ground_truth_sentence in ground_truth_sentences:
            ground_truth_ngrams.extend(get_chunks(ground_truth_sentence))
        pred_ngrams = get_chunks(predicted_sentence)
        count=0
        for i in ground_truth_ngrams:
            if i.startswith('turn'):
                count+=1
        if count>1:
            for i in ground_truth_ngrams:
                if i.startswith('turn'):
                    ground_truth_ngrams.remove(i)
            for i in pred_ngrams:
                if i.startswith('turn'):
                    pred_ngrams.remove(i)

        # use set, don't care about number of ngram occurrences
        ngram_intersection = list(set(ground_truth_ngrams) & set(pred_ngrams))
        percent_match = (len(ngram_intersection)+1e-15) / (len(set(pred_ngrams))+1e-8)

        return percent_match