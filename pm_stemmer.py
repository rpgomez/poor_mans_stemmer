"""This module provides a class for generating a dictionary
of word -> stem from a list of vocabulary words.

To use it, 

1. instantiate an object of class stemmer;

2. use the fit method with a list of vocabulary words; it will return
a dictionary of word -> stem

3. to use it with new content call the stem method with a word and it
 will try to find a stem for the word.

CAVEATS: make sure the vocabulary is consistently lowercase or all
uppercase. The code assumes the case has already been set prior to
use.

"""

import numpy as np
import scipy.stats
from collections import Counter
from collections import defaultdict
from tqdm.auto import tqdm # So we can have a progress bar in console/notebook automagically

def compute_alphabet_prob(words):
    """Recovers the letters used in words and computes the MLE
    probability of letters occurring.  Returns the probability
    distribution as a dictionary with key value pairs (letter,
    Pr(letter)).

    """
    counts = Counter([x for x in "".join(words)])
    total = sum(list(counts.values()))
    probs = {x:counts[x]/total for x in counts}
    return probs

def compute_log_probs(letter_probs):
    """Computes the log likelihood log(Pr(letter)) from a dictionary of
    (letter,Pr(letter)).  Returns the result as dictionary of (letter,
    log(Pr(letter))).

    """
    
    log_probs = {x:np.log(letter_probs[x]) for x in letter_probs}
    return log_probs

def compute_prob_contexts(contexts,log_probs):
    """Computes probs of all observed contexts of length r.
    
    contexts is a list of strings of the same length.
    log_probs is the dictionary of (letter, log(Pr(letter))) pairs.

    returns a dictionary of key value pairs (context, Pr(context))."""
    
    logprobs = {context: sum([log_probs[x] for x in context]) for context in contexts}
    max_log = max(list(logprobs.values()))
    logprobs = {x:logprobs[x] - max_log for x in logprobs}
    probs = {x:np.exp(logprobs[x]) for x in logprobs}
    total = sum(list(probs.values()))
    probs = {x:probs[x]/total for x in probs}
    return probs

def find_suffixes_tighter(vocabulary, log_probs,size=1,debug=False,cutoff = 0.01):
    """Finds all candidate suffixes of size 'size'. 

    vocabulary is the list of words in the vocabulary.
    log_probs is the dictionary of (letter, log(Pr(letter))) pairs.
    size is the size of the suffixes we're looking for.
    debug is a boolean to help me debug this code if needed.
    cutoff is the p value cutoff we're using for hypothesis testing. Default value is 0.01.

    returns a dictionary of (suffix, Pr(suffix)) pairs.
    
    """
    
    suffix_counts  = Counter([x[-size:] for x in vocabulary if len(x) > size])
    suffixes = list(suffix_counts.keys())
    T = len(suffixes)
    if T == 1:
        return {}
    
    probs = compute_prob_contexts(suffixes,log_probs)
    
    n = sum(list(suffix_counts.values()))
    cutoff = cutoff/T # Bonferroni correction
    scores = { x: scipy.stats.binom(n,probs[x]).sf(suffix_counts[x]) for x in suffixes }
    if debug:
        return scores
    candidates = {x: scores[x] for x in scores if scores[x] < cutoff}
    return candidates

def find_suffixes(vocabulary,size=1,cutoff=0.01):
    """Finds all candidate suffixes of size size. 

    vocabulary is the list of words in the vocabulary.
    size is the size of the suffixes we're looking for.
    cutoff is the the p value cutoff we're using for hypothesis testing. Default value is 0.01.

    returns a dictionary of (suffix, Pr(suffix)) pairs.

    The difference between find_prefixes and find_prefixes_tighter is
    that this code assumes the probability of letters is uniform,
    while find_prefixes_tighter does not.

    """
    
    suffix_counts  = Counter([x[-size:] for x in vocabulary if len(x) > size])
    suffixes = list(suffix_counts.keys())
    T = len(suffixes)
    if T == 1:
        return {}
    p = 1/T
    n = sum(list(suffix_counts.values()))
    model = scipy.stats.binom(n,p)
    cutoff = cutoff/T # Bonferroni correction
    scores = { x: model.sf(suffix_counts[x]) for x in suffixes }
    candidates = {x: scores[x] for x in scores if scores[x] < cutoff}
    return candidates

def stripme(aword, suffixes):
    """Attempts to find possible stems for the word.

    aword is a word we're trying to stem.

    suffixes is a dictionary of (length, dictionary2) pairs where
    dictionary2 is a dictionary of (suffix, Pr(suffix)) pairs to
    consider.

    returns a list of possible stems.

    """

    possible_stem = [aword]
    L = len(aword)
    for size in suffixes:
        if size >= L:
            continue

        alist = list(suffixes[size].keys())
        possile_suffix = aword[-size:]
        if  possile_suffix in alist:
            possible_stem.append(aword[:-size])

    return possible_stem

def find_my_best_stem(aword,strong_candidates,mapper):
    """Takes a word and tries to find the best stem for it among a strong list of candidate
    stem words.

    aword is a word we're trying to map to a possible stem in strong_candidates.

    strong_candidates is a list of words that has been vetted as our current best guesses for
    stem words.

    mapper is a dictionary of (word,list of possible stems).

    returns best guess for stem for aword.

    """
    
    cands = [ (len(astem),astem) for astem in strong_candidates if aword in mapper[astem]]
    if len(cands) == 0:
        return aword

    cands.sort()
    best_cand = cands[0][1] # Find the shortest candidate
    return best_cand



class Stemmer():
    def __init__(self):
        """ My constructor. Nothing to do here. """
        self.stem_dict = {}
        self.vocabulary = []
        self.possible_suffixes = {}
        return

    def fit(self,vocabulary,suffix=True,cutoff=0.01,tight=True,verbose=False):
        """Takes a list of vocabulary words and determines a dictionary that
        maps words to their probable stem. This code is not designed to do
        a perfect job, but a pretty good job of assigning words to stems,
        hence the reason it's called Poor Man's Stemmer.

        suffix is a boolean variable that determines if we're trying to detect
        suffixes (True) or prefixes (False). 

        cutoff is the p value cutoff we're using for hypothesis testing.
        
        tight is a boolean variable deciding which of 2 algorithms to use
        for recovering putative suffixes: find_prefixes_tighter (True), find_prefixes (False).

        verbose is a boolean variable for enabling verbose output to track progress. 
        Default is false.
        """

        if not verbose:
            def progress (x,desc=None):
                return x
        else:
            progress = tqdm
        letter_probs = compute_alphabet_prob(vocabulary)
        letter_log_probs = compute_log_probs(letter_probs)

        max_size = max([len(x) for x in vocabulary])

        # find all candidate suffixes sorted on length.
        if tight:
            possible_suffixes = {size:find_suffixes_tighter(vocabulary,letter_log_probs,size=size,\
                                debug=False,cutoff=cutoff) for size in \
                                progress(range(1, max_size),desc='Looking for suffixes')}
        else:
            possible_suffixes = {size:find_suffixes(vocabulary,size=size,cutoff=cutoff)\
                                for size in progress(range(1, max_size),\
                                                     desc='Looking for suffixes')}

        # remove any lengths that have 0 candidate suffixes.
        bad_keys = [x for x in possible_suffixes if len(possible_suffixes[x]) == 0]

        for x in bad_keys:
            temp = possible_suffixes.pop(x) 
        
        self.possible_suffixes = possible_suffixes
        
        # First stab at creating an inverse stem mapper:
        # dictionary (possible stem, list of words)
        self.inverse_map = defaultdict(list) 
        for aword in progress(vocabulary,desc='First pass stemmer map'):
            possible = stripme(aword,possible_suffixes)
            for poss in possible:
                self.inverse_map[poss].append(aword)

        # Now let's see which words have possibly non-trivial stems
        lookatme = [cand for cand in self.inverse_map \
                    if len(self.inverse_map[cand]) >1 and len(cand)> 3]
        self.strong_candidates = list(set(lookatme).intersection(vocabulary))

        self.stem_dict = dict([(aword,find_my_best_stem(aword,self.strong_candidates,
                                                   self.inverse_map)) \
                          for aword in progress(vocabulary,desc='building stem dictionary')])

        return

    def stem(self,aword):
        """Finds the stem for a word"""

        if self.stem_dict == {}:
            # I don't have a stemming dictionary
            return aword

        if aword in self.stem_dict:
            return self.stem_dict[aword]

        # this word was not in my vocabulary.
        # Let's see if we can find something suitable.
        possible_stems = stripme(aword,self.possible_suffixes)
        if len(possible_stems) == 1:
            return aword

        possible_stems = [(len(word), word) for aword in possible_stems \
                          if aword in self.vocabulary]
        possible_stems.sort()

        if len(possible_stems) == 0:
            return aword
        
        best_guess = possible_stems[0][1]
        if len(best_guess) < 4:
            # This stem is too short I think.
            return aword

        return best_guess
