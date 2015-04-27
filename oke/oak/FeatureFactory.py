'''
Created on 24 Apr 2015

@author: jieg
'''
from nltk import word_tokenize
from nltk import pos_tag
import base64

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
        self.dataProcessor = NIF2RDFProcessor()
         
        from nltk.stem import WordNetLemmatizer
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.stoplist=set()
        from nltk.corpus import stopwords
        #The union operator is much faster than add
        self.stoplist |= set(stopwords.words('english'))
        self.stoplist |= set(self.read_by_line('stoplist'))
        
        #load gazetteers
        self.gaz_country=set()
        self.gaz_country |= set(self.read_by_line('gazetteer/country_lower.lst'))
        
        self.gaz_countryAdj=set()
        self.gaz_countryAdj |= set(self.read_by_line('gazetteer/country_adj.lst'))
        #normalise
        self.gaz_countryAdj = set(map(str.lower,self.gaz_countryAdj))
        
        self.gaz_loc_key = set()
        self.gaz_loc_key |= set(self.read_by_line('gazetteer/loc_key.lst'))
        self.gaz_loc_key |= set(self.read_by_line('gazetteer/loc_prekey.lst'))
        self.gaz_loc_key |= set(self.read_by_line('gazetteer/street.lst'))
        #normalise
        self.gaz_loc_key = set(map(str.lower,self.gaz_loc_key))
        
        self.gaz_org_key = set()
        self.gaz_org_key |= set(self.read_by_line("gazetteer/org_base.lst"))
        self.gaz_org_key |= set(self.read_by_line("gazetteer/org_ending.lst"))
        self.gaz_org_key |= set(self.read_by_line("gazetteer/org_key.lst"))
        self.gaz_org_key |= set(self.read_by_line("gazetteer/org_pre.lst"))
        self.gaz_org_key |= set(self.read_by_line("gazetteer/govern_key.lst"))
        #normalise
        self.gaz_org_key = set(map(str.lower,self.gaz_org_key))
        
        self.gaz_person_name=set()
        self.gaz_person_name |= set(self.read_by_line("gazetteer/person_female.lst"))
        self.gaz_person_name |= set(self.read_by_line("gazetteer/person_female_lower.lst"))
        self.gaz_person_name |= set(self.read_by_line("gazetteer/person_first.lst"))
        self.gaz_person_name |= set(self.read_by_line("gazetteer/person_male.lst"))
        #normalise
        self.gaz_person_name = set(map(str.lower,self.gaz_person_name))
        
        self.gaz_person_title=set()
        self.gaz_person_title |= set(self.read_by_line("gazetteer/title.lst"))
        self.gaz_person_title |= set(self.read_by_line("gazetteer/title_female.lst"))
        self.gaz_person_title |= set(self.read_by_line("gazetteer/title_lower.lst"))
        self.gaz_person_title |= set(self.read_by_line("gazetteer/title_male.lst"))
        self.gaz_person_title |= set(self.read_by_line("gazetteer/title_mil.lst"))
        #normalise
        self.gaz_person_title = set(map(str.lower,self.gaz_person_title))
        
        self.gaz_job_title=set()
        self.gaz_job_title |= set(self.read_by_line("gazetteer/jobtitles.lst"))
        self.gaz_job_title = set(map(str.lower,self.gaz_job_title))
        
        self.gaz_facility_key=set()
        self.gaz_facility_key |= set(self.read_by_line("gazetteer/facility_key.lst"))
        self.gaz_facility_key = set(map(str.lower,self.gaz_facility_key))
        print("gazetteers are loaded!")
        
    
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
        #words, contextURI, previousLabel, position
        if type(context_data) is not TaskContext:
            raise Exception('Type error: context_data must be the instance of oke.oak.TaskContext')
        
        #currentWord=words
        
        context_words=word_tokenize(context_data.isString)
        tagged_context=pos_tag(context_words)
        sem_tagged_context=self.sem_tag(context_words,context_data)
        
        #print('sem_tagged_context:',sem_tagged_context)
        datums=[]
        
        #compute features for each word
        #use sliding window to observe on both left and right hand side
        currentIndex=0
        sliding_window_prev_n_words=5
        sliding_window_next_n_words=3
        
        for tagged_word in tagged_context:
            currentWord=tagged_word[0]
            currentWord_label='O' if sem_tagged_context[currentIndex][1] !='class' else 'class'
            datum = Datum(context_data.contextURI,currentWord,currentWord_label)
            
            datum.previousLabel=datums[currentIndex-1].label if (currentIndex-1) in range(0,len(datums)) else 'None'
            
            features={}
            #word-level features (part-of-speech, case, punctuation,digit,morphology)
            #features["word"]= currentWord
            features["word_pos"]=tagged_word[1]
            features["word_root"]=self.wordnet_lemmatizer.lemmatize(currentWord, pos='n')
            features["is_title"]=str(currentWord).istitle()
            features['all_capital']=currentWord.isupper()
            features["is_word_root_be"]='Y' if self.wordnet_lemmatizer.lemmatize(currentWord, pos='v') == 'be' else 'N'
            features['is_punct_comma']='Y' if str(currentWord) == ',' else 'N'
            features['word_with_digits']='Y' if tagged_word[1]!='CD' and self.contains_digits(str(currentWord)) else 'N'         
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
            
            for last_i in range(1,sliding_window_prev_n_words):
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
            for next_i in range(1, sliding_window_next_n_words):
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
    
    def sem_tag(self, context_words, context_data):
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
       
    def read_by_line(self, filePath):
        """
        load file content
        return file content in list of lines
        """
        with open(filePath,mode='r',encoding='utf8') as f:
            content = [l for l in (line.strip() for line in f) if l]
        return content 
    
    def contains_digits(self, d):
        import re
        _digits = re.compile('\d')
        return bool(_digits.search(d))   
    
if __name__ == '__main__':
    print("Testing Feature Factory...")
    from oke.oak.nif2rdfProcessor import NIF2RDFProcessor
    dataProcessor=NIF2RDFProcessor()
    context_data=dataProcessor.aggregate_context_data(dataProcessor.graphData_goldstandards, 
                                                      'http://www.ontologydesignpatterns.org/data/oke-challenge/task-2/sentence-93#char=0,179',
                                                      'The Southern Intercollegiate Athletic Conference is a College athletic conference consisting of historically black colleges and universities located in the southern United States.')
    featFactory = FeatureFactory()
    datums=featFactory.compute_features(context_data)
    featFactory.writeData(datums,'trainWithFeatures')
    
    datums = featFactory.readData('trainWithFeatures.json')
    train_set = [(datum.features, datum.label) for datum in datums]
    print(train_set)
    
    from nltk.classify.maxent import MaxentClassifier
    
    me_classifier = MaxentClassifier.train(train_set)
    predit_label=me_classifier.classify({'word': 'conference', 'word_root': 'conference', 'word_pos': 'NN', 'isEntity': 'N', 'isStopWord': 'N', 'prev_word_isStopWord': 'N'})
    print('predicted label:',predit_label)
    print('========Show top 5 most informative features========')
    me_classifier.show_most_informative_features(5)
        