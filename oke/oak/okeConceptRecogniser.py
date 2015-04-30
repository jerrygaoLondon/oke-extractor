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
        
    def MEclassifier_class_extraction(self,compute_feature=False, is_train=True):
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
        
        if is_train:
            self.train(train_set)
        else:
            class_classifier=self.load_classifier_model(classifier_pickled="class_inducer_0.67.m")
        
        return None
    
    def ontology_alignment(self,compute_feature=False, is_train=False):
        '''
        ontology alignment for DOLCE+DnS Ultra Lite classes
            : query for dbpedia rdf types -> wordnet path similarity (is-a taxonomy) matching
        '''
        from oke.oak.FeatureFactory import FeatureFactory
        from oke.oak.util import extract_type_label
        
        featureFactory = FeatureFactory()
        
        import collections
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
        
        contextDict = featureFactory.dataProcessor.get_task_context(featureFactory.dataProcessor.graphData_goldstandards)
        entityset=set()
        dulclassset=set()
        without_duclass_num=0
        
        true_positive=0
        false_positive=0
        true_negative=0
        false_negative=0
        
        for context, context_sent in contextDict.items():
            context_data=featureFactory.dataProcessor.aggregate_context_data(featureFactory.dataProcessor.graphData_goldstandards,context,context_sent)
            
            entity_dbpedia_URI = context_data.entity.taIdentRef
            entityClasses = context_data.entity.isInstOfEntityClasses
            
            labelled_class_type = [entityClass.subClassOf for entityClass in entityClasses]
            print('labelled class type:',labelled_class_type)
            
            entity_class_labels=set([entityClass.anchorOf for entityClass in entityClasses])
            entity_rdftypes = featureFactory.dbpedia_query_rdftypes(entity_dbpedia_URI)
            #if there is dul/d0 class
            #http://www.ontologydesignpatterns.org/ont/d0.owl#Location
            entity_rdf_type_labels=set([extract_type_label(featureFactory.get_URI_fragmentIdentifier(rdftype_uri)) for rdftype_uri in entity_rdftypes])
            
            dulClass=[rdftype for rdftype in entity_rdftypes if self.is_d0_class(rdftype)]
            
            entityset.add(context_data.entity.taIdentRef)
            testset=set()
            if len(dulClass) > 0 and dulClass[0] in featureFactory.dul_ontology_classes.keys():
                dulclassset.add(dulClass[0])
                testset.add(dulClass[0])
            else:
                #'<',entity_dbpedia_URI, 
                without_duclass_num+=1
                print(str(without_duclass_num)+'> do not have dul class')
                
                #
                entity_synset=set()
                entity_synset.update(entity_rdf_type_labels)
                entity_synset.update(entity_class_labels)
                
                aligned_type = self.schema_alignment_by_wordnet(entity_synset,featureFactory.dul_ontology_classes)
                print("string similarity aligned type for [",entity_class_labels,'] is [',aligned_type,']')
                dulclassset.add(aligned_type)
                testset.add(aligned_type)            
                
            print("labelled class type:",labelled_class_type)
            print("predicted class type:",testset)
            if (len(testset) > 0 and len(labelled_class_type) == 0):
                false_positive+=1
            elif (list(testset)[0] == list(labelled_class_type)[0]):
                true_positive+=1
            else:
                false_positive+=1
        
        print('precision:', true_positive/(true_positive+false_positive))
        print('entityset size:', len(entityset))
        print('existing dul class size:', len(dulclassset))            
        
    def schema_alignment_by_headword_string_sim(self, entity_synset, dul_ontology_classes):
        '''
        The intuition is that head-word carry important information about concept.
        Dbpedia classification can enrich more meaningful type sometimes with the same words with the Ontology that needs to be aligned.
        With the set of enriched "keywords" about entity and local schema types, we can iteratively compare the maximum similarity.
        By applying a threshold, we can choose a ontology class with maximum likelihood.
        
        params:
        entity_sim - set() contains representative labels about entity and types mentioned in context
        dul_ontology_classes - dict() contains dul classes and representative labels
        '''
        from oke.oak.util import levenshtein_similarity
        most_similiar_dul_class=dict()
        for entity_label in entity_synset:
            entity_label_headword=entity_label.split(' ')[-1:][0]
            
            for classUri,classLabels in dul_ontology_classes.items():
                max_sim = max([levenshtein_similarity(entity_label_headword,class_label) for class_label in classLabels])
                
                if most_similiar_dul_class.get(classUri) is None or most_similiar_dul_class.get(classUri) < max_sim:                    
                    most_similiar_dul_class[classUri]=max_sim
                    
        import operator
        print(most_similiar_dul_class)
        suggested_class= max(most_similiar_dul_class, key=most_similiar_dul_class.get)
        print("suggested_class:",suggested_class)
        suggested_class_prob = most_similiar_dul_class.get(suggested_class)
        
        return suggested_class if suggested_class_prob> 0.9 else None
        
        
    def schema_alignment_by_wordnet(self, entity_synset, dul_ontology_classes):
        '''
        wordnet is-a taxonomy path similarity for alignment
        
        The intuition is that head-word carry important information about concept.
        Dbpedia classification can enrich more meaningful type sometimes with the same words with the Ontology that needs to be aligned.
        With the set of enriched "keywords" about entity and local schema types, we can iteratively compare the maximum similarity.
        By applying a threshold, we can choose a ontology class with maximum likelihood.
        
        params:
        entity_sim - set() contains representative labels about entity and types mentioned in context
        dul_ontology_classes - dict() contains dul classes and representative labels
        '''
        from oke.oak.util import wordnet_shortest_path
        most_similiar_dul_class=dict()
        for entity_label in entity_synset:
            entity_label_headword=entity_label.split(' ')[-1:][0]            
            for classUri,classLabels in dul_ontology_classes.items():
                max_sim = max([wordnet_shortest_path(entity_label_headword,class_label) for class_label in classLabels])
                
                if most_similiar_dul_class.get(classUri) is None or most_similiar_dul_class.get(classUri) < max_sim:                    
                    most_similiar_dul_class[classUri]=max_sim
                    
        import operator
        #choose a most similar one
        suggested_class= max(most_similiar_dul_class, key=most_similiar_dul_class.get)
        suggested_class_prob = most_similiar_dul_class.get(suggested_class)
        
        return suggested_class if suggested_class_prob> 0.0 else None        
            
    def is_d0_class(self,class_uri):
        from urllib.parse import urlparse
        parsedURI = urlparse(class_uri)
        netloc=parsedURI.netloc
        if 'www.ontologydesignpatterns.org' == netloc:
            return True
        return False
           
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
        
        print('========Show top 10 most informative features========')
        classifier.show_most_informative_features(10)
        
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
    conceptRecogniser.MEclassifier_class_extraction(compute_feature=False)
    #conceptRecogniser.ontology_alignment()
    '''
    from oke.oak.FeatureFactory import FeatureFactory
    featureFactory=FeatureFactory()
    entity_synset=['Person','Natural Person','Athlete','Organism','Player',"Volleyball Player"]
    suggested_class = conceptRecogniser.schema_alignment_by_wordnet(entity_synset, featureFactory.dul_ontology_classes)
    print('suggested_class: ',suggested_class)
    '''