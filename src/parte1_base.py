"""Person 1 code, copied verbatim from "Parte 1_testes.ipynb" (cells 1-5 and the
split creation of cell 6) so that Part 2 uses exactly the same data preparation,
models, metrics and 5 group-aware splits. Only changes (marked "(changed)"):
where train.csv is looked for, and loading it only if it exists.
"""
import csv
import json
from pathlib import Path
import pandas as pd
from collections import Counter
import re
import unicodedata
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from itertools import product
from sklearn.dummy import DummyClassifier
from time import perf_counter
from threadpoolctl import threadpool_limits

# (changed) look for train.csv in StudentsPack/, the project folder or the current folder
_here = Path(__file__).resolve().parent
folder = next((f for f in [_here.parent / 'StudentsPack', _here.parent, Path.cwd()] if (f / 'train.csv').exists()), _here.parent / 'StudentsPack')

csv_fields = ['description', 'sample_name', 'keywords', 'transcription']

specialties = ['Cardiovascular-Pulmonary', 'Dermatology', 'Gastroenterology','General Medicine', 'Neurology', 'Neurosurgery', 'Obstetrics-Gynecology','Ophthalmology', 'Orthopedic', 'Psychiatry-Psychology', 'Radiology', 'Surgery',]

def load_train():
    return pd.read_csv(folder / 'train.csv', sep=';', keep_default_na=False)

def load_test(path):
    return pd.read_csv(path, sep=';', names=['description', 'sample_name', 'transcription', 'keywords'], keep_default_na=False)

study_folder = Path(__file__).resolve().parent.parent / 'Test'
study_folder.mkdir(parents=True, exist_ok=True)


def prepare_data(data):

    #creates copy to not alter the original Dataframe.
    data = data.copy().reset_index(drop=True)
    
    #Keeps only the specified 12 specialties; if there is an extra one or it is a random word it is excluded
    data = data[data['medical_specialty'].isin(specialties)].reset_index(drop=True)

    #Converts NAN with ' ' and turns it into strings
    for field in csv_fields:
        data[field] = data[field].fillna('').astype(str)

    #Creates unique id. This is useful due to the partitions in train and validation
    if 'row_id' not in data:
        data['row_id'] = np.arange(len(data))

    #Group cases with identical descriptions or transcriptions to keep each group in one partition, this prevents duplicate text from appearing
    #both in training and validation
    if 'group_id' not in data:

        #Assigns each row to its own group
        parents = list(range(len(data)))

        #Finds the representative of the group with this row
        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        
        #Checks descriptions and transcriptions. Before comparing texts we normaliza case and whitespace
        for field in ['description', 'transcription']:
            seen = {}
            for index, text in enumerate(data[field].map(normalize)):
                if not text:
                    continue
                if text in seen:
                    parents[find(index)] = find(seen[text])
                else:
                    seen[text] = index

        #Save the final group representative for each note. Note: this detects exact normalized matches, not approximate similarity
        data['group_id'] = [find(index) for index in range(len(data))]
        
    return data

def inspect_data(data):

    #Sumarizes the training data so we can analyse odd cases and report it
    summary = {
        'cases': len(data), #number of cases after preparation and filtering
        'classes': data['medical_specialty'].value_counts().to_dict(), #Number of cases in each medical specialty
        'groups': int(data['group_id'].nunique()), #Number of distinct groups to keep duplicate cases together
        'empty_fields': {field: int(data[field].str.strip().eq('').sum()) for field in csv_fields}, #Pretty self explanatory, cases where its empty
        'keywords_labels_removed': int((data['keywords'] != prepare_texts(data)['keywords']).sum()), #Pretty self explanatory also
        'test_used': False,}

#We normalize the text by converting it into a string, then caracteres, then lowercasing, trading various spaces into just one and removing spaces
#if theyre at the beggining and end of the text
def normalize(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    return re.sub(r'\s+', ' ', text).strip()


#I noticied various patterns with the transcription. It includes subsections like HISTORY, DIAGNOSIS, PLANS... this function is just to recognise
#the various subsections and create a list of them.
def select_sections(text):
    markers = list(re.finditer(r'(?:^|[,\n])\s*([A-Z][A-Z /()-]{1,65}?)\s*:{1,2}', str(text)))
    parts = []
    for i, marca in enumerate(markers):
        if any(name in marca.group(1) for name in ['DIAGNOS', 'IMPRESSION', 'ASSESSMENT', 'PROCEDURE', 'FINDINGS', 'PLAN']):

            #if there is another title, it stops before it. Otherwise, if it is, it ends on the end of the text 
            end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
            
            #puts everythig together and removes ',', spaces, and so on
            parts.append(text[marca.start():end].lstrip(',\n '))
    return '\n'.join(parts)


#Pretty self explanatory, prepares the texts. Creates a copy and replaces missing values with ' '. Also garantees that all 4 fields have text
def prepare_texts(data, keywords='strip', sections=False, concatenate=False, fields=None):
    texts = data.copy()
    for field in csv_fields:
        texts[field] = texts[field].fillna('').astype(str)

    #Normalizes the specialties names. For example, i noticied that Cardiovascular-Pulmonary is written this way, with a -, so we take it off and 
    #replace it with a space, becoming cardiovascular pulmonary. its useful because we can compare those names with the first keyword
    names = set()
    for name in specialties:
        names.add(normalize(re.sub(r'[/\-]', ' ', name)))

    #The objective here is to avoid that a keyword, like surgery, reveals directly the specialty that the model should predict.     
    keyword_values = []
    
    for text in texts['keywords']:
        first = text.strip().split(',', 1)[0]
        first = normalize(re.sub(r'[/\-]', ' ', first))
        if keywords != 'raw' and first in names:
            text = text.split(',', 1)[1].strip() if ',' in text else ''
        keyword_values.append(text)
    texts['keywords'] = keyword_values

    #This mode deletes all keywords to evaluate how the model performs without this content
    if keywords == 'none':
        texts['keywords'] = ''

    #The mode mask does the same BUT besides that also searches for the specialty names in the 4 fields and builds patterns to look for those names
    #with slight variations, for example spaces. In the end, replaces each name found with the word specialty. 
    if keywords == 'mask':
        patterns = [re.escape(name).replace(r'\-', r'\s*[/\-]?\s*').replace(r'\ ', r'\s+') for name in specialties]
        pattern = re.compile(r'(?<!\w)(?:' + '|'.join(patterns) + r')(?!\w)', re.I)
        for field in csv_fields:
            texts[field] = texts[field].map(lambda t: pattern.sub('specialty', t))

    #To choose if we use the selections-subsections or not, depending on the model configuration we choose
    if sections:
        texts['transcription'] = texts['transcription'].map(select_sections)

    #To choose if we use concatenate, aka merging columns text. Example; merging description and sample_name
    if concatenate:
        texts['flat'] = texts[fields or csv_fields].agg(' '.join, axis=1)
    return texts




#Can and will most likely have slight alterations, this is just for testing. 
#For the default, i chose all fields, in my mind it makes sense to feed the model with all the information available
#For the representation i also chose both because it combines subwording-level and character-level and helps to capture clinical terms better
#Although i proposed a logistic regression in the initial plan, I'm gonna experiment with SVM https://www.reddit.com/r/AskStatistics/comments/krl8sg/how_would_you_explain_the_difference_between_an/
#I still have doubts with the c value. A lower level penalizes more the complexity of the model. Still needs validation
#I chose to strip the first keyword since it usually reveals the specialty, but we can test with more cases
#Unchangable note weight
#Chose char weight 0.5 to treat characters more as additional information rather than main one (like words)
#I chose unigrams and bigrams, we can test variations but would let it as it is

def create_model(fields=None, representation='both', classifier='svm', c=0.5, keywords='strip', note_weight=1.0, char_weight=0.5, ngrams=2, concatenate=False, sections=False):

    #As said before, we choose all 4 fields
    if fields is None:
        fields = csv_fields

    #Appends the vectors that will transform each field into numbers. Weights, as the name indicates, appends the weight applied to each field.     
    transformations = []
    block_weights = {}

    
    for field in (['flat'] if concatenate else fields):

        #Here, we lowercase the text, apply sublinear to mitigate repeated words,require that the term appears 2 times and limit the vocabulary size. Also defines what counts as a word
        #Disclaimer: AI helped me identify which number i should choose for max_features
        words = TfidfVectorizer(lowercase=True, ngram_range=(1, ngrams), sublinear_tf=True, min_df=2 if field == 'transcription' else 1, max_features=30000 if field == 'transcription' else 10000, token_pattern=r"(?u)\b\w+(?:[./'-]\w+)*\b", dtype=np.float64,)

        
        #We apply this vector when using words
        name = field + '_word'
        if representation in ['word', 'both']:
            transformations.append((name, words, field))
            block_weights[name] = note_weight if field == 'transcription' else 1.0

        
        #Pretty much the same thing, but here we transform subwords into numbers. Each subword needs to appear at least in 2 documents
        if representation in ['char', 'both'] and (field != 'transcription' or representation == 'char'):
            characters = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=13000, dtype=np.float64,)
            name = field + '_char'
            transformations.append((name, characters, field))
            block_weights[name] = (char_weight if representation == 'both' else 1.0) * (note_weight if field == 'transcription' else 1.0)

    
    #Here we apply all the vectors created into the respective columns to have the results in a single matrix
    tfidf = ColumnTransformer(transformations, transformer_weights=block_weights, sparse_threshold=1.0, remainder='drop', verbose_feature_names_out=False,)
    
    #As i said before, i wanted to experiment with SVM, but just insert any other value if you want the logistic regression
    if classifier == 'svm':

        algorithm = LinearSVC(C=c, dual='auto', max_iter=20000, random_state=42)
    else:
        #I just chose 1000 iteractions here but its changable
        algorithm = LogisticRegression(C=c, solver='lbfgs', max_iter=1000, random_state=42)
        
    return Pipeline([
        ('cleaning', FunctionTransformer(prepare_texts, kw_args={'keywords': keywords, 'sections': sections, 'concatenate': concatenate, 'fields': fields,})),
        ('tfidf', tfidf), ('classifier', algorithm),])


#Adjust class weight for classes that dont exist as often.
def class_weights(labels, q=1):
    counts = Counter(labels)
    weights = {}
    for label in specialties:
        weights[label] = (len(labels) / (len(specialties) * counts[label])) ** q
    mean = np.mean([weights[label] for label in labels])
    return np.array([weights[label] / mean for label in labels])


#Stemmming, will be useful for later
def stemmed_words(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(p) for p in re.findall(r'(?u)\b\w\w+\b', text.lower())]

    
#Simulate baseline thats described on the project proposal
def create_baseline():
    tfidf = TfidfVectorizer(lowercase=True, tokenizer=stemmed_words, token_pattern=None, ngram_range=(1, 1), sublinear_tf=False, min_df=1, dtype=np.float64,)
    
    return Pipeline([('cleaning', FunctionTransformer(prepare_texts)),('tfidf', ColumnTransformer([('description_word', tfidf, 'description')], sparse_threshold=1.0,verbose_feature_names_out=False,)), ('classifier', MultinomialNB()),])

#Here we calculate the metrics. Pretty self explanatory.
#p corresponds to precision, r for recall, f for f1 and n for Support
    
#https://www.evidentlyai.com/classification-metrics/multi-class-metrics
#Accuracy: Proportion of correct predictions
#Macro_precision: Average precision across all specialties
#macro_recall: Average recall across all specialties
#macro_f1: Average F1 score across all specialties, giving equal weight to each
#weighted_f1: F1 score averaged and weighted by the number of cases in each specialty
#min_f1: The lowest F1 score among the specialties
#meets_thresholds: Whether accuracy is greater than 0.49 and all F1 scores are at least 0.25
#per_class:Individual results for each specialty

def measure(truth, predictions, weights=None):
    p, r, f, n = precision_recall_fscore_support(truth, predictions, labels=specialties, zero_division=0, sample_weight=weights)
    
    accuracy = accuracy_score(truth, predictions, sample_weight=weights)
    return {'accuracy': float(accuracy), 'macro_precision': float(np.mean(p)), 'macro_recall': float(np.mean(r)), 'macro_f1': float(np.mean(f)), 
            'weighted_f1': float(f1_score(truth, predictions, labels=specialties, average='weighted', zero_division=0, sample_weight=weights)),
            'min_f1': float(np.min(f)), 'meets_thresholds': bool(accuracy > .49 and np.min(f) >= .25), 
            'per_class': pd.DataFrame({'specialty': specialties, 'precision': p,
            'recall': r, 'f1': f, 'support': n}).to_dict('records'),}

#According to the teacher, we need to achieve a baseline of 0.49 at least in accuracy and 0.25 in metrics. It returns the values so we can compare models
def priority(metrics):
    violation = max(0, .49 - metrics['accuracy']) + max(0, .25 - metrics['min_f1'])
    return (metrics['meets_thresholds'], -violation, metrics['macro_f1'], metrics['accuracy'])


#Creates folder with the results
def save_evaluation(directory, data, predictions, folds, times):

    #If the folder exist it wont be created
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    #Compares the previsions with the real specialties and calculates the global metrics
    summary = measure(data['medical_specialty'], predictions)

    #Keeps the prevision of each case, to identify the errors and the division where each case was evaluated
    table = data[['row_id', 'group_id', 'medical_specialty']].copy()
    table['prediction'], table['fold'] = predictions, folds
    table.to_csv(directory / 'predictions.csv', index=False)

    #Saves metrics per specialty
    pd.DataFrame(summary.pop('per_class')).to_csv(directory / 'classes.csv', index=False)
    cm = confusion_matrix(data['medical_specialty'], predictions, labels=specialties)

    #Saves the confusion matrixes
    for name, matrix in [('confusao', cm), ('confusao_normalizada', cm / cm.sum(axis=1, keepdims=True))]:
        pd.DataFrame(matrix, index=specialties, columns=specialties).to_csv(directory / f'{name}.csv')
        
    rows = []

    #Folds divide the data into subsets so the model can be trained and validated on different cases, so we can also verify consistency
    #We save each folders metrics and time
    for fold in sorted(set(folds)):
        indices = np.array(folds) == fold
        m = measure(data.loc[indices, 'medical_specialty'], np.array(predictions)[indices])
        rows.append({'fold': int(fold), **{k: v for k, v in m.items() if k != 'per_class'}, 'seconds': times[fold]})
        
    fold_table = pd.DataFrame(rows)
    fold_table.to_csv(directory / 'folds.csv', index=False)

    summary['folds_meeting_thresholds'] = int(fold_table['meets_thresholds'].sum())
    summary['seconds'] = float(sum(times))
    summary['fold_mean'] = fold_table[['accuracy', 'macro_f1', 'min_f1']].mean().to_dict()
    summary['fold_std'] = fold_table[['accuracy', 'macro_f1', 'min_f1']].std().to_dict()
    weights = 1 / data['group_id'].map(data['group_id'].value_counts())
    summary['equal_group_weight_metrics'] = {k: v for k, v in measure(data['medical_specialty'], predictions, weights).items() if k != 'per_class'}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return {k: v for k, v in summary.items() if not isinstance(v, dict)}





# (changed) only needed for training/evaluation; predicting with a saved model works without train.csv
data = prepare_data(load_train()) if (folder / 'train.csv').exists() else None

#Create five reproducible splits, keeping duplicate groups together and attempting to preserve specialty proportions
if data is not None:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    splits = [{'fold': fold, 'train': data.iloc[train_indices]['row_id'].tolist(), 'evaluation': data.iloc[validation_indices]['row_id'].tolist(),}
        for fold, (train_indices, validation_indices) in enumerate(splitter.split(data, data['medical_specialty'], groups=data['group_id']))]
