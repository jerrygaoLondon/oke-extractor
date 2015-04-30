'''
Created on 28 Apr 2015

@author: jieg
'''
def levenshtein_similarity(str1,str2):
        '''
        Implements the basic Levenshtein algorithm providing a similarity measure between two strings
        return actual / possible levenstein distance to get 0-1 range normalised by the length of the longest sequence
        '''
        #sim_score=self.load_sim_from_memory(str1, str2)
        #if sim_score is None:
        from distance import nlevenshtein
        dist=nlevenshtein(str1, str2, method=1)
        sim_score= 1 - dist
        
        return sim_score

def wordnet_shortest_path(word1, word2):
    from nltk.corpus import wordnet as wn
    word1_synsets=wn.synsets(word1)
    word2_synsets=wn.synsets(word2)
    
    sem_similarity=0.0
    
    for word1_synset in word1_synsets:
        for word2_synset in word2_synsets:
            _similarity=word1_synset.path_similarity(word2_synset)
            if _similarity is not None and sem_similarity < _similarity:
                sem_similarity = _similarity
    #print('semantic similarity between [',word1,'] and [',word2,'] is [',sem_similarity,']')
    return sem_similarity

def extract_type_label(ident_str):
    '''
    extract_type_label_for_dbpedia_uri_fragmentIdentifier
    before applying the semantic similarity method,
        identifier is transformed and normalised into a meaningful form that can be easily recognised.
    quick solution to extract (esp. for yago ontology):
        FictionalCharacter109587565 -> Fictional Character
        District108552138->District
        Location100027167->Location
    alternative way is to query via endpoint or ontology (e.g., yago)
    '''
    tokens=[]
    _temp_token=''
    for i in range(0,len(ident_str)-1):
        cur_char=ident_str[i]
        if i!=0 and (cur_char.isupper() or cur_char.isdigit()):
            if _temp_token != '':
                tokens.append(_temp_token)
            _temp_token=''
            if cur_char.isdigit() is False:
                _temp_token=cur_char
        else:
            _temp_token+=cur_char
    
    return ' '.join(tokens)
    