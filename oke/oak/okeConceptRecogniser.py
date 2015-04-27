'''
Created on 23 Apr 2015

@author: jieg
'''
import sys
sys.path.append("../../")

class ConceptRecogniser(object):
    '''
    Maximum entropy based classifier for NER trained on goldstandard sample data provided by oke-challenge
    '''

    stoplist=set()
    def __init__(self):
        '''
        Constructor
        '''
        if len(self.stoplist) == 0 :
            from nltk.corpus import stopwords
            #The union operator is much faster than add
            self.stoplist |= set(stopwords.words('english'))
            self.stoplist |= set(self.read_by_line('stoplist'))
        
    def MEclassifier_class_extraction(self,compute_feature=False):
        '''
        Maximum entropy based classifier for Class Induction
        '''
        from oke.oak.FeatureFactory import FeatureFactory
        #load the train (goldstandard) data
        featureFactory = FeatureFactory()
        if compute_feature:
            #compute and export training features
            featureFactory.export_to_features('trainWithFeatures')
        else:
            print("skip computing features...")
        
        print('load features from \'trainWithFeatures.json\'... ')
        # write the updated data into JSON files
        datums = featureFactory.readData('trainWithFeatures.json')
        train_set = [(datum.features, datum.label) for datum in datums]
        print("train set size",len(train_set))
        
        self.train(train_set)
        
        return None

    def train(self, train_set):
        split_size_train=0.7
        print(' split ',split_size_train*100,'% from gold standards for training ... ')
        
        from nltk.classify.maxent import MaxentClassifier       
        
        import random
        random.shuffle(train_set)
        
        _train_set, _test_set=train_set[:round(len(train_set)*split_size_train)],train_set[round(len(train_set)*split_size_train):]
        me_classifier = MaxentClassifier.train(_train_set)
        
        self.benchmarking(me_classifier,_test_set)
         
        self.save_classifier_model(me_classifier,'class_inducer.m')

    def benchmarking(self, classifier,_test_set):
        from nltk import classify
        accuracy = classify.accuracy(classifier, _test_set)
        print("accuracy:",accuracy)
        
        from nltk.metrics import precision
        from nltk.metrics import recall
        from nltk.metrics import f_measure
        
        import collections
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        for i, (feats, label) in enumerate(_test_set):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
            
        print('precision:', precision(refsets['class'], testsets['class']))
        print('recall:', recall(refsets['class'], testsets['class']))
        print('F-measure:', f_measure(refsets['class'], testsets['class']))        
        
        print('========Show top 6 most informative features========')
        classifier.show_most_informative_features(6)
        
    def save_classifier_model(self, model, outfile):
        import pickle
        if model:
            with open(outfile, 'wb') as model_file:
                pickle.dump(model, model_file)
    
    def load_classifier_model(self, classifier_pickled = None):
        import pickle
        if classifier_pickled:
            print("Load trained model from", classifier_pickled)
            with open(classifier_pickled, 'rb') as model:
                classifier = pickle.load(model)
            return classifier
        
    def entity_rel_pattern_matching(self, context_sentence, entity_name):
        '''
        PoS tagging based entity Is-A type definition rule extraction
        return a list of tuple
        '''
        rel_grammar='rel: {<VBZ|VBD><RB>?<DT>+<VBG>?<JJ>*<NN|NNP|NNS>+<IN>?<DT>?<NN|NNP|NNS>*}'

        #Year 23 BC was either a common year starting on Saturday or Sunday or a leap year starting on Friday
        #The DFB-Pokal 1980-81 was the 38th season of the competition
        #Alain de Lille, theologian and poet, was born in Lille, some years before 1128.
        #A bicycle, often called a bike or cycle, is a human-powered, pedal-driven, single-track vehicle,
        class_name_pos_filters=['JJ','VBG','NN','NNP','NNS']
        rel_indicator_filters=['be']
        class_grammars=['class: {<NN|NNP|NNS>+}','class: {<JJ>+ <NN|NNP|NNS>+}']
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        
        import nltk
        relParser = nltk.RegexpParser(rel_grammar)
        
        #linguistic pre-processing context sentence
        sent_tokens = nltk.word_tokenize(context_sentence)
        tagged_context_sent = nltk.pos_tag(sent_tokens)
        
        #pattern matching for IS-A relation
        rel_chunk=relParser.parse(tagged_context_sent)
        rel_context_list=[]
        for node_a in rel_chunk:
            if type(node_a) is nltk.Tree:
                if node_a.label() == 'rel':
                    #filter relation ship
                    rel_indicator_word=node_a[0][0]
                    rel_indicator_root=wordnet_lemmatizer.lemmatize(rel_indicator_word,pos='v')
                    if rel_indicator_root not in rel_indicator_filters:
                        continue
                    
                    rel_context_list.append(list(node_a))
                        
        print("[%s] is-A relationship found" % len(rel_context_list))
        
        #Class Induction from IS-A relation       
        rels=set()
        for relContext in rel_context_list:
            len_rel_context=len(relContext)
            headWord_index=len_rel_context-1
            #add head word as class first
            if (relContext[headWord_index][0] not in self.stoplist):
                rels.add(relContext[headWord_index][0])                
            
            preword_index=headWord_index-1
            
            while (relContext[preword_index][1] in class_name_pos_filters) and (relContext[preword_index][0] not in self.stoplist):
                comb_context=list(w for (w,pos) in relContext[preword_index:headWord_index+1])                
                rels.add(' '.join(comb_context))
                preword_index=preword_index-1
        entity_rels = [(entity_name,is_a) for is_a in rels]
        return entity_rels
    
    def read_by_line(self, filePath):
        """
        load file content
        return file content in list of lines
        """
        DELIMITER = "\n"
        with open(filePath) as f:
            content = [line.rstrip(DELIMITER) for line in f.readlines()]
            
        return content

if __name__ == '__main__':
    print("test Maximum Entropy classifer for class induction")
    #from oke.oak.okeConceptRecogniser import ConceptRecogniser
    conceptRecogniser=ConceptRecogniser()
    conceptRecogniser.MEclassifier_class_extraction(compute_feature=True)