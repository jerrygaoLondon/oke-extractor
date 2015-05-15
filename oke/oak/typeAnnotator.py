'''
Created on 8 May 2015

@author: jieg
'''
import sys
sys.path.append("../../")

class TypeAnnotator(object):
    '''
    classdocs
    '''

    ns_prefix_oke="http://www.ontologydesignpatterns.org/data/oke-challenge/task-2/"
    
    def __init__(self):
        '''
        Constructor
        '''
      
    def type_annotation(self,graph_in_memory, context_data, predicted_class_tagged_expressions):
        '''
        Given identify the type(s) of the given entity as they are expressed in the given definition, 
            (ii) create a owl:Class statement for defining each of them as a new class in the target knowledge base, 
            (iii) create a rdf:type statement between the given entity and the new created classes,
        param:
        graph_in_memory, rdf graph data in memory (load by oke.oak.NIF2RDFProcessor)
        context_data, TaskContext
        predicted_class_tagged_expressions, tuple list, pos tagged class phrases [(word,pos),...]
        
        '''
        multiple_type_expression_set=set()
        for predicted_class_tagged_expression in predicted_class_tagged_expressions:
            multiple_type_expressions= self.extract_multiple_type_expression(predicted_class_tagged_expression)
            multiple_type_expression_set.update(multiple_type_expressions)
        
        #generate taIdentRef URI dict
        multiple_type_expressions_uris={type_expression: self.generate_uri(self.ns_prefix_oke,type_expression) for type_expression in multiple_type_expression_set}
        print("generated uris:",multiple_type_expressions_uris)
        
        context_uri=context_data.contextURI
        context_content=context_data.isString
        entity_taIdentRef=context_data.entity.taIdentRef
        entity_label=context_data.entity.anchorOf
        
        for _type_expression in multiple_type_expressions_uris:
            #if _type_expression is None:
            #    continue
            
            type_beginIndex,type_endIndex=self.get_char_index(context_content, _type_expression)
            
            type_expression_uri=self.generate_type_expression_uri(context_uri, type_beginIndex, type_endIndex)
            self.annotate_type_expression_graph(graph_in_memory, 
                                                type_expression_uri, 
                                                _type_expression, 
                                                type_beginIndex, 
                                                type_endIndex, 
                                                context_uri, 
                                                multiple_type_expressions_uris[_type_expression])
            self.annotate_entity_with_type(graph_in_memory, entity_taIdentRef,entity_label, multiple_type_expressions_uris[_type_expression])
            self.annotate_type_as_class(graph_in_memory, multiple_type_expressions_uris[_type_expression], _type_expression)
        
        print_for_test=graph_in_memory.serialize(format='n3')
        print("print_for_test for updated graph: ",print_for_test)
    
    def type_alignment_annotation(self, graph_in_memory, context_data, suggested_dul_alignments):
        print("annotate type alignments")
        entityTypes = context_data.entity.isInstOfEntityClasses
        
        if entityTypes:
            from rdflib import RDFS
            from rdflib.term import URIRef
            
            for entityType in entityTypes:
                entityType_taIdentRef = entityType.taIdentRef
                print("entityType_taIdentRef:", entityType_taIdentRef)
                for suggested_dul_class in suggested_dul_alignments:
                    graph_in_memory.add((URIRef(entityType_taIdentRef), RDFS.subClassOf,
                                         URIRef(suggested_dul_class)))
            graph_in_memory.commit()
            
            
    def annotate_type_expression_graph(self,graph_in_memory,type_expression_uri, type_expression, type_beginIndex,type_endIndex,context_uri, _taIdentRef):
        from rdflib.term import URIRef
        from rdflib.term import Literal
        from rdflib import RDF
        from rdflib.namespace import XSD
        print("annotate type expression in graph...")
        graph_in_memory.add((URIRef(type_expression_uri), RDF.type, 
                            URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#String')))
        graph_in_memory.add((URIRef(type_expression_uri), RDF.type, 
                            URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#RFC5147String')))
        graph_in_memory.add((URIRef(type_expression_uri), URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#anchorOf'),
                            Literal(type_expression, lang='en')))
        graph_in_memory.add((URIRef(type_expression_uri), URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#beginIndex'),
                            Literal(type_beginIndex, datatype=XSD.integer)))
        graph_in_memory.add((URIRef(type_expression_uri), URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#endIndex'),
                            Literal(type_endIndex, datatype=XSD.integer)))
        graph_in_memory.add((URIRef(type_expression_uri), URIRef('http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#referenceContext'),
                            URIRef(context_uri)))
        graph_in_memory.add((URIRef(type_expression_uri),URIRef('http://www.w3.org/2005/11/its/rdf#taIdentRef'),
                            URIRef(_taIdentRef)))
        
        graph_in_memory.commit()
        
        #print_for_test=graph_in_memory.serialize(format='n3')
        #print("print_for_test for updated graph: ",print_for_test)
    def annotate_entity_with_type(self,graph_in_memory,entity_taIdentRef,entity_label, type_taIdentRef):
        from rdflib.term import URIRef
        from rdflib import RDF
        from rdflib import RDFS
        from rdflib.term import Literal
        
        print("entity taIdentRef:",entity_taIdentRef)
        graph_in_memory.add((URIRef(entity_taIdentRef), RDF.type, URIRef(type_taIdentRef)))
        graph_in_memory.add((URIRef(entity_taIdentRef), RDFS.label, Literal(entity_label, lang='en')))
        graph_in_memory.commit()
        
    def annotate_type_as_class(self, graph_in_memory,  type_taIdentRef, type_label):
        from rdflib.term import URIRef
        from rdflib import RDF
        from rdflib import RDFS
        from rdflib.term import Literal
        
        graph_in_memory.add((URIRef(type_taIdentRef), RDF.type, URIRef('http://www.w3.org/2002/07/owl#Class')))
        graph_in_memory.add((URIRef(type_taIdentRef), RDFS.label, Literal(type_label, lang='en')))
        graph_in_memory.commit()
        
    def generate_type_expression_uri(self, context_uri, type_beginIndex, type_endIndex):
        '''
        type rdf information related to context sentence
        e.g., <http://www.ontologydesignpatterns.org/data/oke-challenge/task-2/sentence-67#char=37,49>
        '''
        from urllib.parse import urlparse
        parsed_context_URI = urlparse(context_uri)
        type_expression_uri=parsed_context_URI.netloc+parsed_context_URI.path+"#char="+str(type_beginIndex)+","+str(type_endIndex)
        return type_expression_uri
        
    def get_char_index(self,context_content, type_expression):
        '''
        get char index range of type phrases corresponding to original context content
        (18, 35)
        '''        
        return (context_content.index(type_expression),context_content.index(type_expression)+len(type_expression))
        
    def generate_uri(self,namespace="", anchorLabel=""):
        from urllib.parse import quote
        
        return namespace+quote("".join([w.capitalize() for w in anchorLabel.split(' ')]))
                                                                    
    def extract_multiple_type_expression(self, predicted_class_expressions):
        '''
        param:
        predicted_class: tuple list , pos tagged class phrases [(word,pos),...]
        
        return a set of string expression
        '''
        multiple_type_expressions_tagged_list = self.extract_multiple_type_expressions_tagged_list(predicted_class_expressions)
        
        multiple_type_expression_set=set()
        multiple_type_expression_set.add(" ".join([word for (word, pos) in predicted_class_expressions]))
        multiple_type_expression_set.add(self.get_head_noun(predicted_class_expressions))
        for extracted_tagged_type in multiple_type_expressions_tagged_list:
            multiple_type_expression_set.add(" ".join([word for (word, pos) in extracted_tagged_type]))
            
        return multiple_type_expression_set
        
    def extract_multiple_type_expressions_tagged_list(self, predicted_class_expressions):
        class_grammars=['class: {<JJ>+ <NN|NNP|NNS>+}', 
                'class: {<VBG>+ <NN|NNP|NNS>+}',
                'class:{<VBD>+ <NN|NNP|NNS>+}', 
                'class:{<NN|NNP|NNS> <JJ>+ <NN|NNP|NNS>+}',
                'class:{<NN|NNP|NNS> <VBD>+ <NN|NNP|NNS>+}', 
                'class:{<NN|NNP|NNS> <VBG>+ <NN|NNP|NNS>+}',
                'class:{<NN|NNP|NNS>+ <VBD>+ <NN|NNP|NNS>+}', 
                'class:{<NN|NNP|NNS>+ <VBG>+ <NN|NNP|NNS>+}', 
                'class: {<NN|NNP|NNS>+ <VBG>* <VBD>* <JJ>* <NN|NNP|NNS>+}']
        
        print("extract_multiple_type_expressions_tagged_list for [ ", predicted_class_expressions)
        import nltk
        multiple_type_expressions_tagged_list=list()
        
        for class_grammar in class_grammars:
            relParser = nltk.RegexpParser(class_grammar)
            rel_chunk=relParser.parse(predicted_class_expressions)
            
            for node_a in rel_chunk:
                if type(node_a) is nltk.Tree:
                    if node_a.label() == 'class':
                        multiple_type_expressions_tagged_list.append(list(node_a))
        #print("multiple type_expressions tagged list:", multiple_type_expressions_tagged_list)
        return multiple_type_expressions_tagged_list
    
    def get_head_noun(self, predicted_class_expressions):
        '''
        get head noun from pos tagged phrases
        '''
        if predicted_class_expressions:
            return predicted_class_expressions[-1][0] if predicted_class_expressions[-1][1] in ['NN','NNS','NNP'] else ""
        return ""
    
    def test_extract_multiple_type_expressions_tagged_list(self):
        predicted_class_expressions=[('fictional', 'JJ'), ('villain', 'NN')]
        self.extract_multiple_type_expressions_tagged_list(predicted_class_expressions)
        print("head noun:", self.get_head_noun(predicted_class_expressions))
        
        predicted_class_expressions=[('common', 'NN'), ('year', 'NN')]
        self.extract_multiple_type_expressions_tagged_list(predicted_class_expressions)
        
        #League Baseball Left Fielder
        predicted_class_expressions=[('League', 'NNP'), ('Baseball', 'NNP'), ('Left', 'VBD'), ('Fielder','NN')]
        self.extract_multiple_type_expressions_tagged_list(predicted_class_expressions)
        
        predicted_class_expressions=[('season', 'NN'), ('of', 'IN'), ('the', 'DT'), ('competition','NN')]
        self.extract_multiple_type_expressions_tagged_list(predicted_class_expressions)

    def test_extract_multiple_type_expression(self):
        predicted_class_expressions=[('fictional', 'JJ'), ('villain', 'NN')]        
        print("extracted type expressions", self.extract_multiple_type_expression(predicted_class_expressions))
        
        predicted_class_expressions=[('season', 'NN'), ('of', 'IN'), ('the', 'DT'), ('competition','NN')]        
        print("extracted type expressions", self.extract_multiple_type_expression(predicted_class_expressions))
        
        predicted_class_expressions=[('League', 'NNP'), ('Baseball', 'NNP'), ('Left', 'VBD'), ('Fielder','NN')]        
        
        print("extracted type expressions", self.extract_multiple_type_expression(predicted_class_expressions))
    
        testExpressions = self.extract_multiple_type_expression(predicted_class_expressions)
        multiple_type_expressions_uris={type_expression: self.generate_uri(self.ns_prefix_oke,type_expression) for type_expression in testExpressions}
                
        print(multiple_type_expressions_uris)
        
    def test_generate_uri(self):
        generatedURI=self.generate_uri(self.ns_prefix_oke, "All's Well That Ends Well")
        print(generatedURI)
        
if __name__ == '__main__':
    typeAnnotator=TypeAnnotator()
    typeAnnotator.test_extract_multiple_type_expression()