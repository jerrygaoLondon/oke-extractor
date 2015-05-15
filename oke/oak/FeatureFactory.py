'''
Created on 24 Apr 2015

@author: jieg
'''
from nltk import word_tokenize
from nltk import pos_tag

import sys
sys.path.append("../../")
from oke.oak.TaskContext import TaskContext
from oke.oak.Datum import Datum
import json

class FeatureFactory(object):
    '''
    classdocs
    '''
    dataProcessor=None
    wordnet_lemmatizer=None
            
    def __init__(self):
        '''
        initialise NIF2RDF Data processor and load training data
        '''
        from oke.oak.nif2rdfProcessor import NIF2RDFProcessor
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        from oke.oak.util import read_by_line
        
        self.dataProcessor = NIF2RDFProcessor()         
        
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.stoplist=set()
        
        #The union operator is much faster than add
        self.stoplist |= set(stopwords.words('english'))
        self.stoplist |= set(read_by_line('stoplist'))
        
        #load gazetteers
        #Gazetteer and trigger word features for relation extraction
        self.gaz_country=set()
        self.gaz_country |= set(read_by_line('gazetteer/country_lower.lst'))
        
        self.gaz_countryAdj=set()
        self.gaz_countryAdj |= set(read_by_line('gazetteer/country_adj.lst'))
        #normalise
        self.gaz_countryAdj = set(map(str.lower,self.gaz_countryAdj))
        
        self.gaz_loc_key = set()
        self.gaz_loc_key |= set(read_by_line('gazetteer/loc_key.lst'))
        self.gaz_loc_key |= set(read_by_line('gazetteer/loc_prekey.lst'))
        self.gaz_loc_key |= set(read_by_line('gazetteer/street.lst'))
        #normalise
        self.gaz_loc_key = set(map(str.lower,self.gaz_loc_key))
        
        self.gaz_org_key = set()
        self.gaz_org_key |= set(read_by_line("gazetteer/org_base.lst"))
        self.gaz_org_key |= set(read_by_line("gazetteer/org_ending.lst"))
        self.gaz_org_key |= set(read_by_line("gazetteer/org_key.lst"))
        self.gaz_org_key |= set(read_by_line("gazetteer/org_pre.lst"))
        self.gaz_org_key |= set(read_by_line("gazetteer/govern_key.lst"))
        #normalise
        self.gaz_org_key = set(map(str.lower,self.gaz_org_key))
        
        self.gaz_person_name=set()
        self.gaz_person_name |= set(read_by_line("gazetteer/person_female.lst"))
        self.gaz_person_name |= set(read_by_line("gazetteer/person_female_lower.lst"))
        self.gaz_person_name |= set(read_by_line("gazetteer/person_first.lst"))
        self.gaz_person_name |= set(read_by_line("gazetteer/person_male.lst"))
        #normalise
        self.gaz_person_name = set(map(str.lower,self.gaz_person_name))
        
        self.gaz_person_title=set()
        self.gaz_person_title |= set(read_by_line("gazetteer/title.lst"))
        self.gaz_person_title |= set(read_by_line("gazetteer/title_female.lst"))
        self.gaz_person_title |= set(read_by_line("gazetteer/title_lower.lst"))
        self.gaz_person_title |= set(read_by_line("gazetteer/title_male.lst"))
        self.gaz_person_title |= set(read_by_line("gazetteer/title_mil.lst"))
        #normalise
        self.gaz_person_title = set(map(str.lower,self.gaz_person_title))
        
        self.gaz_job_title=set()
        self.gaz_job_title |= set(read_by_line("gazetteer/jobtitles.lst"))
        self.gaz_job_title = set(map(str.lower,self.gaz_job_title))
        
        self.gaz_facility_key=set()
        self.gaz_facility_key |= set(read_by_line("gazetteer/facility_key.lst"))
        self.gaz_facility_key = set(map(str.lower,self.gaz_facility_key))
        print("gazetteers are loaded!")
        
    def is_matched(self, _string, gaz_set):
        '''
        return True if _string is matched by one of element in gaz_set
        '''
        first_matched=[gaz_str for gaz_str in gaz_set if gaz_str in _string.lower().split(' ') or _string.lower() in gaz_str.split(' ')]
        return len(first_matched) >0
        
    def export_to_features(self,json_file):
        print("start to load training data and compute training features...")
        
        contextDict = self.dataProcessor.get_task_context(self.dataProcessor.graphData_goldstandards)
        total_training_size=len(contextDict)
        
        print("Total [%s] context sentence in training data"%total_training_size)
        
        processed_context_num=0
        for context, context_sent in contextDict.items():
            context_data=self.dataProcessor.aggregate_context_data(self.dataProcessor.graphData_goldstandards,context,context_sent)
            datums=self.compute_features(context_data)
            self.writeData(datums,json_file,'a')
            
            processed_context_num+=1
            current_progress=round((processed_context_num/total_training_size),2)*100
            print('current progress',current_progress,'%')
        print("complete!")
        
    def writeData(self, data, filename, mode='a'):
        '''
        write Datum data into external file with option 'a' - append and 'w' - overwrite
        '''
        outFile = open(filename + '.json', mode,encoding='utf8')
        for i in range(0, len(data)):
            datum = data[i]
            jsonObj = {}
            jsonObj['_contextURI']=datum.contextURI
            jsonObj['_word']=datum.word
            jsonObj['_label']=datum.label
            
            featureObj = datum.features
            jsonObj['_features'] = featureObj
            
            outFile.write(json.dumps(jsonObj) + '\n')
        outFile.close()
    
    def readData(self, filename):
        data = [] 
        import codecs
        with codecs.open(filename,'rU','utf-8') as f:
            for line in f:
                json_data = json.loads(line)
            
                word = json_data['_word']
                label = json_data['_label']
                contextURI=json_data['_contextURI']
                
                datum = Datum(contextURI, word, label)
                datum.features=json_data['_features']
                
                data.append(datum)
        return data
                 
    def compute_features(self, context_data):
        '''
        Maximum entropy model gives a better performance for sequence labelling problem. 
        By maximizing the entropy in our model, we are attempting to minimise the amount of the information the model carries.
        Design a language model to maximise the entropy and 
            feed our language model with a set of features associated with a given token we wish to classify
            and the system can then given us the probability that our token falls into any given class of token against which our language model was trained.
        '''
        from oke.oak.util import wordnet_shortest_path
        from oke.oak.util import extract_type_label
        from oke.oak.util import get_URI_fragmentIdentifier
        from oke.oak.util import contains_digits
        
        #words, contextURI, previousLabel, position
        if type(context_data) is not TaskContext:
            raise Exception('Type error: context_data must be the instance of oke.oak.TaskContext')
        
        context_words=word_tokenize(context_data.isString)
        tagged_context=pos_tag(context_words)
        sem_tagged_context=self.sem_tag(context_words,context_data)        
        
        entity_name=context_data.entity.anchorOf
        entity_head_word=entity_name.split(' ')[-1:][0]
        entity_dbpedia_URI = context_data.entity.taIdentRef
        #print("entity_dbpedia_URI:"+entity_dbpedia_URI)
        '''
        LOD based semantic type feature:
        '''
        entity_rdftypes=self.entity_rdftypes_feature_extraction(entity_dbpedia_URI)         
        
        if (len (entity_rdftypes) == 0):
            print("Warn: No rdf types can be found for [current word")#entity_name.decode("utf8"),"]")
        # extract labels from RDF type
        entity_semantics=set()
        entity_semantics.update(set([extract_type_label(get_URI_fragmentIdentifier(rdftype_uri)) for rdftype_uri in entity_rdftypes]))
        
        #print('sem_tagged_context:',sem_tagged_context)
        #add head word into rdf type
        #  to avoid adding head word into rdf type: not many head word represent essential word associated with type
        #entity_semantics.add(entity_head_word)
        #print("entity_semantics:",entity_semantics)
        datums=[]
        
        #compute features for each word
        #use sliding window to observe on both left and right hand side
        currentIndex=0
        sliding_window_prev_n_words=8
        sliding_window_next_n_words=3
        
        for tagged_word in tagged_context:
            currentWord=tagged_word[0]
            #label encoding
            currentWord_label='O' if sem_tagged_context[currentIndex][1] !='class' else 'class'
            datum = Datum(context_data.contextURI,currentWord,currentWord_label)
            
            datum.previousLabel=datums[currentIndex-1].label if (currentIndex-1) in range(0,len(datums)) else 'None'
            
            features={}
            #word-level features (part-of-speech, case, punctuation,digit,morphology)
            import string
            if currentWord.lower() not in self.stoplist and currentWord not in string.punctuation and currentWord.isdigit() is not True and tagged_word[1] in ["NN", "NNP", "NNS"]:
                #use lemmatised word
                features["word"]= self.wordnet_lemmatizer.lemmatize(currentWord, pos='n')
                #Word sense of Noun: we can use "WN_CLASS" to determine whether the NN word is a hyponym of w (or keywords) in ontology by wordnet
                #features["WN_CLASS"]=
            features["word_pos"]=tagged_word[1]
            #features["word_root"]=self.wordnet_lemmatizer.lemmatize(currentWord, pos='n')
            features["is_title"]=str(currentWord).istitle()
            features['all_capital']=currentWord.isupper()
            features["is_word_root_be"]='Y' if self.wordnet_lemmatizer.lemmatize(currentWord, pos='v') == 'be' else 'N'
            features['is_punct_comma']='Y' if str(currentWord) == ',' else 'N'
            features['word_with_digits']='Y' if tagged_word[1]!='CD' and contains_digits(str(currentWord)) else 'N'         
            features["is_StopWord"]='Y' if currentWord in self.stoplist else 'N'
            features["is_Entity"]='N' if sem_tagged_context[currentIndex] !='entity' else 'Y'
            features["last_2_letters"]='None' if len(str(currentWord))<=2 or str(currentWord).isdigit() else str(currentWord)[-2:]
            #type_indicator can be retrieved by wordnet synonyms
            features["type_indicator"]='Y' if currentWord in ['name','form','type','class','category', 'variety', 'style','model','substance', 'version', 'genre','matter','mound', 'kind', 'shade', 'substance'] else 'N'
            
            #semantic (gazetteer lookup) features
            features["is_orgKey"] ='Y' if currentWord.lower() in self.gaz_org_key else 'N'
            features["is_locKey"] = 'Y' if currentWord.lower() in self.gaz_loc_key else 'N'
            features["is_country"] = 'Y' if currentWord.lower() in self.gaz_country else 'N'
            features["is_countryAdj"]='Y' if currentWord.lower in self.gaz_countryAdj else 'N'
            features["is_personName"] = 'Y' if currentWord.lower() in self.gaz_person_name else 'N'
            features["is_personTitle"] = 'Y' if currentWord.lower() in self.gaz_person_title else 'N'
            features['is_jobtitle']='Y' if currentWord.lower() in self.gaz_job_title else 'N'
            features['is_facKey']='Y' if currentWord.lower() in self.gaz_facility_key else 'N'
            
            #add feature to compute path similarity between dbpedia type and current word
            
            if entity_semantics:
                max_sim = max([wordnet_shortest_path(currentWord,sem_type.split(' ')[-1:][0]) for sem_type in entity_semantics])
                features['sim_dist_with_DbpediaType'] = max_sim
            
            
            for last_i in range(1,sliding_window_prev_n_words+1):
                if currentIndex == 0:
                    features['prev_word']="<START>"
                
                if currentIndex != 0 and currentIndex-last_i >=0:                    
                    #features['prev_'+str(last_i)+'_word']=datums[currentIndex-last_i].features['word'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_pos']=datums[currentIndex-last_i].features['word_pos'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    #features['prev_'+str(last_i)+'_word_root']=datums[currentIndex-last_i].features['word_root'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'                
                    features['prev_'+str(last_i)+'_word_is_StopWord']=datums[currentIndex-last_i].features['is_StopWord'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'         
                    features['prev_'+str(last_i)+'_word_is_Entity']=datums[currentIndex-last_i].features['is_Entity'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_title']=datums[currentIndex-last_i].features['is_title'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_all_capital']=datums[currentIndex-last_i].features['all_capital'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_word_root_be']=datums[currentIndex-last_i].features['is_word_root_be'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_punct_comma']=datums[currentIndex-last_i].features['is_punct_comma'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_word_with_digits']=datums[currentIndex-last_i].features['word_with_digits'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_last_2_letters']=datums[currentIndex-last_i].features['last_2_letters'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_type_indicator']=datums[currentIndex-last_i].features['type_indicator'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_orgKey']=datums[currentIndex-last_i].features['is_orgKey'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_locKey']=datums[currentIndex-last_i].features['is_locKey'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_country']=datums[currentIndex-last_i].features['is_country'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_countryAdj']=datums[currentIndex-last_i].features['is_countryAdj'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_personName']=datums[currentIndex-last_i].features['is_personName'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_personTitle']=datums[currentIndex-last_i].features['is_personTitle'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                    features['prev_'+str(last_i)+'_word_is_facKey']=datums[currentIndex-last_i].features['is_facKey'] if (currentIndex-last_i) in range(0,len(datums)) else 'None'
                
            datum.features=features
            currentIndex+=1
            datums.append(datum)
        
        #add features about next words
        #reset to 0
        currentIndex = 0
        for tagged_word in tagged_context:
            for next_i in range(1, sliding_window_next_n_words+1):
                if ((currentIndex+next_i) == len(datums)):
                    datums[currentIndex].features['next_word']="<END>"
                
                if (currentIndex+next_i) != len(datums) :
                    #datums[currentIndex].features['next_'+str(next_i)+'_word']=datums[currentIndex+next_i].features['word'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_pos']=datums[currentIndex+next_i].features['word_pos'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_StopWord']=datums[currentIndex+next_i].features['is_StopWord'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_Entity']=datums[currentIndex+next_i].features['is_Entity'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'

                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_title']=datums[currentIndex+next_i].features['is_title'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_all_capital']=datums[currentIndex+next_i].features['all_capital'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_word_root_be']=datums[currentIndex+next_i].features['is_word_root_be'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_punct_comma']=datums[currentIndex+next_i].features['is_punct_comma'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_word_with_digits']=datums[currentIndex+next_i].features['word_with_digits'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_last_2_letters']=datums[currentIndex+next_i].features['last_2_letters'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_type_indicator']=datums[currentIndex+next_i].features['type_indicator'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_orgKey']=datums[currentIndex+next_i].features['is_orgKey'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_locKey']=datums[currentIndex+next_i].features['is_locKey'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_country']=datums[currentIndex+next_i].features['is_country'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_countryAdj']=datums[currentIndex+next_i].features['is_countryAdj'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_personName']=datums[currentIndex+next_i].features['is_personName'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_personTitle']=datums[currentIndex+next_i].features['is_personTitle'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
                    datums[currentIndex].features['next_'+str(next_i)+'_word_is_facKey']=datums[currentIndex+next_i].features['is_facKey'] if (currentIndex+next_i) in range(0,len(datums)) else 'None'
            currentIndex+=1
            
        return datums
    
    def entity_rdftypes_feature_extraction(self, entity_dbpedia_URI):
        '''
        Extracting entity RDF type as feature from DBpedia
        
        return set : entity rdf type set
        '''
        import SPARQLWrapper
        from oke.oak.DBpediaQuerier import DBpediaQuerier
        dbpediaQuerier = DBpediaQuerier()
        try:
            entity_rdftypes = dbpediaQuerier.dbpedia_query_rdftypes(entity_dbpedia_URI,endpoint="http://galaxy.dcs.shef.ac.uk:8893/sparql")
        except SPARQLWrapper.SPARQLExceptions.EndPointInternalError:
            print("EndPointInternalError: try default dbpedia endpoint next...")
        entity_rdftypes.update(dbpediaQuerier.dbpedia_query_rdftypes(entity_dbpedia_URI))
        #print("entity_rdftypes:",entity_rdftypes)
        return entity_rdftypes
    
    def sem_tag(self, context_words, context_data):
        '''
        tag context words/tokens with semantic tags in gold standards (i.e.,entity and class expressions)
        params:
        context_words : list of tokens by word_tokenize
        context_data: TaskContext
        '''
        semantic_info_dict=[(context_data.entity.anchorOf,'entity')]+[(entityClass.anchorOf, 'class') for entityClass in context_data.entity.isInstOfEntityClasses]
        return self.dictionary_tagging(context_words, semantic_info_dict)

    def dictionary_tagging(self, tokens, _dict):
        '''
        sequentially tag tokens with the given dictionary (multiple tuple list)
        '''
        #initialise
        #sem_tagged_tokens=[(token,None) for token in tokens]
        token_sem_index={}
        max_token_len=len(tokens)
        for dict_tuple in _dict:
            dict_name=dict_tuple[0]
            dict_name_tag=dict_tuple[1]
            dict_name_tokens=dict_name.split(' ')
            name_len=len(dict_name_tokens)
            for i in range(0,max_token_len):
                beginIndex=i
                endIndex=i+name_len
                
                if(endIndex>max_token_len):
                    break
                
                if (tokens[beginIndex:endIndex] == dict_name_tokens):
                    token_sem_index.update({index:dict_name_tag for index in range(beginIndex,endIndex)})
                i+=1
        
        sem_tagged_tokens=[(tokens[i],None if i not in token_sem_index else token_sem_index[i]) for i in range(0, max_token_len)]
        return sem_tagged_tokens        
            
    '''
    ========================================Simple test/evaluation methods below ===========================================================
    '''
        
    def test_feature_extraction_for_maxent_classifier(self):
        print("Testing Feature extraction for maxent classifier...")
        from oke.oak.nif2rdfProcessor import NIF2RDFProcessor
        dataProcessor=NIF2RDFProcessor()
        context_data=dataProcessor.aggregate_context_data(dataProcessor.graphData_goldstandards, 
                                                          'http://www.ontologydesignpatterns.org/data/oke-challenge/task-2/sentence-93#char=0,179',
                                                          'The Southern Intercollegiate Athletic Conference is a College athletic conference consisting of historically black colleges and universities located in the southern United States.')
        featFactory = FeatureFactory()
        datums=featFactory.compute_features(context_data)
        featFactory.writeData(datums,'test_trainWithFeatures')
        
        datums = featFactory.readData('test_trainWithFeatures.json')
        train_set = [(datum.features, datum.label) for datum in datums]
        print(train_set)
        
        from nltk.classify.maxent import MaxentClassifier
        
        me_classifier = MaxentClassifier.train(train_set)
        predit_label=me_classifier.classify({'word': 'conference', 'word_root': 'conference', 'word_pos': 'NN', 'isEntity': 'N', 'isStopWord': 'N', 'prev_word_isStopWord': 'N'})
        print('predicted label:',predit_label)
        print('========Show top 10 most informative features========')
        me_classifier.show_most_informative_features(10)
    
if __name__ == '__main__':
    featFactory = FeatureFactory()
    featFactory.test_feature_extraction_for_maxent_classifier()