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
    _classifier=None
    
    def __init__(self):
        '''
        Constructor
        '''
        if len(self.stoplist) == 0 :
            from nltk.corpus import stopwords
            #The union operator is much faster than add
            self.stoplist |= set(stopwords.words('english'))
            self.stoplist |= set(self.read_by_line('stoplist'))
    
    def type_extraction_and_interlink(self, nif_content):
        '''
        Type prediction and alignment
        More details refer to J.Gao and S. Mazumdar. Exploiting Linked Open Data to Uncover Entity Types
        
        Input(n3) -> Parsing & Processing -> Feature Extraction -> Feature set -> classifier 
            -> predicted classes -> type annotation -> alignment -> alignment annotation
        return enriched results in n3    
        '''
        print("==========type_extraction_and_interlink============")
        print(type(nif_content))
        if nif_content == "" or nif_content is None:
            raise Exception("No content to extract")
            return
        
        contextDict, graph_in_memory = self.parse_and_processing(nif_content)
        
        from oke.oak.typeAnnotator import TypeAnnotator
        from oke.oak.dulOntologyAligner import DulOntologyAligner
        from oke.oak.nif2rdfProcessor import NIF2RDFProcessor
        
        dataProcessor = NIF2RDFProcessor() 
        typeAnnotator=TypeAnnotator()
        dulOntologyAligner=DulOntologyAligner()
        
        for context, context_sent in contextDict.items():
            datums,context_data = self.feature_extraction_for_prediction(graph_in_memory, context, context_sent)
            
            predicted_classes = self.prediction_on_feature_set(datums)
            
            print("predicted class for current context: ",predicted_classes)
            typeAnnotator.type_annotation(graph_in_memory,context_data, predicted_classes)
            
            #reload graph
            context_data=dataProcessor.aggregate_context_data(graph_in_memory,context,context_sent) 
            suggested_dul_alignments=dulOntologyAligner.ontology_alignment(context_data)
            print("suggested_dul_alignments:", suggested_dul_alignments)
            typeAnnotator.type_alignment_annotation(graph_in_memory, context_data, suggested_dul_alignments)          
        
        return graph_in_memory.serialize(format='n3')
                
    def parse_and_processing(self, nif_content):
        '''
            parsing&processing in prediction phase
        '''
        from oke.oak.nif2rdfProcessor import NIF2RDFProcessor
        dataProcessor = NIF2RDFProcessor()
        graph_in_memory=dataProcessor.load_rdf_from_content(nif_content, _format='n3')
             
        dataProcessor.validate_graph_in_memory(graph_in_memory)
        
        contextDict = dataProcessor.get_task_context(graph_in_memory)
        total_test_size=len(contextDict)
        print("total test context: [",total_test_size,"]")
        
        return (contextDict,graph_in_memory)
    
    def feature_extraction_for_prediction(self, graph_in_memory, context, context_sent):
        '''
        feature extraction for current context task in prediction phase
        '''
        from oke.oak.nif2rdfProcessor import NIF2RDFProcessor
        from oke.oak.FeatureFactory import FeatureFactory
        
        dataProcessor = NIF2RDFProcessor()
        featureFactory = FeatureFactory()
        
        context_data=dataProcessor.aggregate_context_data(graph_in_memory,context,context_sent)
        datums=featureFactory.compute_features(context_data)
        
        return (datums,context_data)
    
    def prediction_on_feature_set(self, datums):
        '''
        Predict type (entity class) based on feature set
        '''
        class_label="class"
        predicted_class=[]
        #load classifier only once
        if not self._classifier:
            self._classifier=self.MEclassifier_model(compute_feature=False,is_train=False)
            
        _temp_class_token=[]
        for datum in datums:
            #(i) identify the type(s) of the given entity as they are expressed in the given definition
            predicted_label = self._classifier.classify(datum.features)
            #print("word ",datum.word,"|pos:",datum.features["word_pos"]," -> predicted class:",predicted_label)            
            #a nasty way to get continuous label, possibly need to change to BIO label
            if predicted_label == class_label:
                _temp_class_token.append((datum.word,datum.features["word_pos"]))
            else:
                if _temp_class_token:
                    predicted_class.append(_temp_class_token)
                    _temp_class_token=[]
        return predicted_class
        
    def MEclassifier_model(self,compute_feature=False, is_train=True):
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
            class_classifier=self.train(train_set)
        else:
            class_classifier=self.load_classifier_model(classifier_pickled="me_class_inducer.m")
        
        return class_classifier
    
           
    def train(self, train_set):
        split_size_train=0.7
        print(' split ',split_size_train*100,'% from gold standards for training ... ')
        
        from nltk.classify.maxent import MaxentClassifier       
        from nltk.classify.naivebayes import NaiveBayesClassifier        
        
        # 10 fold test
        fold_n=2
        all_f_measure=[]
        all_precision=[]
        all_recall=[]
        import random
        for i in range(1,fold_n):
            print("start [%s] fold validation..." %i)
            random.shuffle(train_set)
        
            _train_set, _test_set=train_set[:round(len(train_set)*split_size_train)],train_set[round(len(train_set)*split_size_train):]
        
            me_classifier = MaxentClassifier.train(_train_set)
            #nb_classifier = NaiveBayesClassifier.train(_train_set)

            #from sklearn.svm import LinearSVC
            #from nltk.classify.scikitlearn import SklearnClassifier
            #print("training SVM Classifier...")
            #svm_classifier = SklearnClassifier(LinearSVC())
            #svm_classifier = svm_classifier.train(_train_set)
            #print("complete SVM training.")
        
            self.benchmarking(me_classifier,_test_set,all_f_measure, all_precision, all_recall)
        print("all_f_measure,",all_f_measure)
        print("all_precision,",all_precision)
        print("all_recall", all_recall)
        
        print("Final F-measure", sum(all_f_measure) / float(len(all_f_measure))) 
        print("Final precision", sum(all_precision) / float(len(all_precision))) 
        print("Final recall", sum(all_recall) / float(len(all_recall))) 
        
        self.save_classifier_model(me_classifier,'me_class_inducer.m')
        return me_classifier

    def benchmarking(self, classifier,_test_set,all_f_measure=[],all_precision=[],all_recall=[]):
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
            
        prec=precision(refsets['class'], testsets['class'])
        rec=recall(refsets['class'], testsets['class'])
        f1=f_measure(refsets['class'], testsets['class'])
        print('precision:', prec)
        print('recall:', rec)
        print('F-measure:', f1)
                
        all_f_measure.append(f1)
        all_precision.append(prec)
        all_recall.append(rec)
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

    def test_type_extraction_and_interlink(self):
        test_file_path="../test/task2_test.ttl"
        with open(test_file_path,'r',encoding="utf-8") as f:
            content=f.read()     
        
        returnedN3Graph = self.type_extraction_and_interlink(content)
        with open("../test/task2_test_output.ttl", mode='wb') as f:
            f.write(returnedN3Graph)
        print("returnedN3Graph:")
        print(returnedN3Graph)
if __name__ == '__main__':
    print("test Maximum Entropy classifer for class induction")
    #from oke.oak.okeConceptRecogniser import ConceptRecogniser
    conceptRecogniser=ConceptRecogniser()
    #conceptRecogniser.MEclassifier_model(compute_feature=False)
    #conceptRecogniser.ontology_alignment()
    
    conceptRecogniser.test_type_extraction_and_interlink()
    '''
    from oke.oak.FeatureFactory import FeatureFactory
    featureFactory=FeatureFactory()
    entity_synset=['Person','Natural Person','Athlete','Organism','Player',"Volleyball Player"]
    suggested_class = conceptRecogniser.schema_alignment_by_wordnet(entity_synset, featureFactory.dul_ontology_classes)
    print('suggested_class: ',suggested_class)
    '''