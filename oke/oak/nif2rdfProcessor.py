'''
Created on 23 Apr 2015

@author: jieg
'''
import rdflib

import sys
sys.path.append("../../")

class NIF2RDFProcessor(object):
    '''
    RDF parser and querier for task dataset
    '''
    graphData_goldstandards=None
    def __init__(self):
        '''
        Constructor
        '''       
        self.graphData_goldstandards = self.load_rdf("../GoldStandard_sampleData/dataset_task_2.ttl", 'n3')
        print("training data is loaded!")
    
    def load_D0_ontology(self):
        
        graph_in_memory=self.load_rdf("d0.owl",_format="xml")
        return graph_in_memory
    
    def load_DUL_ontology(self):
        graph_in_memory=self.load_rdf("DUL.owl.xml",_format="xml")
        return graph_in_memory
        
    def is_subClass_of(self, graph_in_memory, class_uri, reference_class_uri):
        '''
        check whether class_uri is subclass of reference_class_uri
        return True or False
        '''
        is_subclass_of_query="ASK { <%s>" %class_uri
        is_subclass_of_query+=" rdfs:subClassOf <%s>" %reference_class_uri
        is_subclass_of_query+="   } "
        
        answerResult = graph_in_memory.query(is_subclass_of_query)
        
        return answerResult.askAnswer
    
    def is_subClassOf_dul_d0_class(self, class_uri, reference_class_uri):
        dul_ontology_in_memory=self.load_DUL_ontology()
        answer=self.is_subClass_of(dul_ontology_in_memory, class_uri,reference_class_uri)
        
        if answer is False:
            d0_ontology_in_memory=self.load_D0_ontology()
            answer=self.is_subClass_of(d0_ontology_in_memory, class_uri,reference_class_uri)
            return answer
        else:
            return answer
                
    def load_rdf(self, filePath, _format='n3'):
        
        goldstandards_file = open(filePath, "r",encoding="utf8")
        
        graph_in_memory = rdflib.Graph("IOMemory")
        
        
        #graph_in_memory.load("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl", "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl", "xml")
        return graph_in_memory.parse(file=goldstandards_file,format=_format)
    
    def load_rdf_from_content(self, rdf_content, _format='n3'):
        graph_in_memory = rdflib.Graph("IOMemory")
        
        return graph_in_memory.parse(data=rdf_content,format=_format)
        
    def query(self, graph_in_memory, sparql_query=""):
        '''
        graph_in_memory is a rdflib.graph.Graph
        return result bindings(a list)
        '''
        self.validate_graph_in_memory(graph_in_memory)
        
        relations=graph_in_memory.query(sparql_query)
        print("result size:", len(relations.bindings))

        return relations.bindings
    
    def validate_graph_in_memory(self, graph_in_memory):
        if type(graph_in_memory) is not rdflib.graph.Graph:
            raise Exception("Only rdflib.graph.Graph is supported!")
        
        if len(graph_in_memory) == 0:
            print("No statement in memory to query!")
            return None
            
    def get_task_context(self,graph_in_memory):
        '''
        query context to process
        return Dict() with sentence id as key and sentence as value
        '''
        if type(graph_in_memory) is not rdflib.graph.Graph:
            raise Exception("Only rdflib.graph.Graph is supported!")
        
        query='select ?s ?context where {?s a nif:Context; nif:isString ?context.}'
        relations=self.query(graph_in_memory, query)
        print("total [%s] result returned" % len(relations))
        
        contextDict= {str(relations[i]['?s']): str(relations[i]['?context']) for i in range(0, len(relations))}
        return contextDict
    
    def get_task_context_entity(self,graph_in_memory, contextId=""):
        '''
        get task context entities (the priori entity) in order
        return a triple tuple list about entities
        ('entityId','anchorOf', 'beginIndex','endIndex','taIdentRef')
        '''
        if type(graph_in_memory) is not rdflib.graph.Graph:
            raise Exception("Only rdflib.graph.Graph is supported!")
        
        query='select distinct ?entityId ?anchorOf ?beginIndex ?endIndex ?taIdentRef where {?entityId a nif:String; nif:anchorOf ?anchorOf; nif:referenceContext <%s>;' %contextId
        query+='nif:beginIndex ?beginIndex; nif:endIndex ?endIndex; itsrdf:taIdentRef ?taIdentRef.'
        #set optitonal here support prediction when we don't have labelled entity type information
        query+='OPTIONAL {?taIdentRef a ?type.}'
        query+='OPTIONAL {FILTER(?type != owl:Class)}'
        query+="} order by ?beginIndex"
        
        relations=self.query(graph_in_memory, query)
        print("total [%s] entities found" % len(relations))
        
        contextRel= [(str(relations[i]['entityId']),str(relations[i]['anchorOf']), str(relations[i]['beginIndex']), 
                      str(relations[i]['endIndex']),str(relations[i]['taIdentRef'])) for i in range(0, len(relations))]
        return contextRel
    
    def get_entity_types(self, graph_in_memory, entity_taIdentRef, referenceContext):
        '''
        get task context entity labelled classes/types
        return a triple tuple list about the labelled entity types
        
        '''
        if type(graph_in_memory) is not rdflib.graph.Graph:
            raise Exception("Only rdflib.graph.Graph is supported!")
        # ?taIdentRef a ?type .
        query='select distinct ?anchorOf ?type ?dulClass ?beginIndex ?endIndex ?refContext ?entityTypeIdentRef'
        query+='where {<%s> a ?type . ?type a owl:Class; rdfs:subClassOf ?dulClass .' %entity_taIdentRef
        
        query+=' ?entityTypeId a nif:String; nif:anchorOf ?anchorOf; nif:beginIndex ?beginIndex;'
        query+=' nif:endIndex ?endIndex; nif:referenceContext ?refContext; itsrdf:taIdentRef ?entityTypeIdentRef. '
        
        query+=' FILTER (str(?refContext) = \'%s\') ' %referenceContext
        query+=' FILTER (?entityTypeIdentRef = ?type)'
        query+='}'
        
        from pyparsing import ParseException
        try:
            relations=self.query(graph_in_memory, query)
        except ParseException:
            raise Exception("ParseException! Pls Check possible encoding or format issue in param [entity_taIdentRef] Current value is [%s]!" %entity_taIdentRef)
             
        entityTypeRel=[(str(relations[i]['anchorOf']),str(relations[i]['type']), str(relations[i]['dulClass']), 
                        str(relations[i]['beginIndex']), str(relations[i]['endIndex']), 
                        str(relations[i]['refContext']),
                        str(relations[i]['entityTypeIdentRef'])) for i in range(0, len(relations))]
        return entityTypeRel
        
    def aggregate_context_data(self,graph_in_memory, contextId="", context_sent=""):
        from oke.oak.TaskContext import TaskContext
        currentContext=TaskContext(contextId)
        #currentContext.contextURI=contextId
        currentContext.isString=context_sent
        entity_info = self.get_task_context_entity(graph_in_memory, contextId)
        from oke.oak.TaskContext import ContextEntity
        
        entity=ContextEntity(entity_info[0][0])
        #entity.entityURI=entity_info[0][0]
        entity.anchorOf=entity_info[0][1]
        entity.beginIndex=entity_info[0][2]
        entity.endIndex=entity_info[0][3]
        entity.taIdentRef=entity_info[0][4]
        
        entity.referenceContext=contextId
        
        entity_labelled_types=self.get_entity_types(graph_in_memory, entity.taIdentRef, contextId)
        from oke.oak.TaskContext import EntityClass
        entityClasses=[]
        for entity_labelled_type in entity_labelled_types:
            entityClass=EntityClass(entity_labelled_type[1])
            entityClass.anchorOf=entity_labelled_type[0]
            #entityClass.uri=entity_labelled_type[1]
            entityClass.subClassOf=entity_labelled_type[2]
            entityClass.beginIndex=entity_labelled_type[3]
            entityClass.endIndex=entity_labelled_type[4]
            entityClass.referenceContext=entity_labelled_type[5]
            entityClass.taIdentRef=entity_labelled_type[6]
            entityClasses.append(entityClass)
        
        entity.isInstOfEntityClasses=entityClasses
        currentContext.entity=entity
        return currentContext        
        
    def get_task_entities_labelledType(self,graph_in_memory):
        return None
    '''
    ============================================Evaluation/Test=======================================================================
    '''
    def test_is_subclass_of(self):  
        dul_ontology_in_memory=self.load_DUL_ontology()
        answer=self.is_subClass_of(dul_ontology_in_memory, "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#NaturalPerson", "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Person")
        print("is \"dul:NaturalPerson\" subClassOf \"dul:Person\":",answer)
    
    def test_is_subClassOf_dul_d0_class(self):
        answer=self.is_subClassOf_dul_d0_class("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#NaturalPerson", "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent")
        print("is 'dul:NaturalPerson' subClassOf 'dul:Agent':", answer)
        
        answer=self.is_subClassOf_dul_d0_class("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Goal", "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Description")
        print("is 'dul:Goal' subClassOf 'dul:Description':", answer)
        
        answer=self.is_subClassOf_dul_d0_class("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Task", "http://www.ontologydesignpatterns.org/ont/d0.owl#Activity")
        print("is 'dul:Task' subClassOf 'd0:Description':", answer)
        
        answer=self.is_subClassOf_dul_d0_class("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Personification", "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#SocialAgent")
        print("is 'dul:Personification' subClassOf 'dul:SocialAgent':", answer)
        
        answer=self.is_subClassOf_dul_d0_class("http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Personification", "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent")
        print("is 'dul:Personification' subClassOf 'dul:Agent':", answer)
    
    def test_load_rdf(self):
        graph_in_memory=self.load_rdf("../test/task2_test.ttl", _format='n3')
        
        print("len of graph:", len(graph_in_memory))
        from rdflib.graph import Graph
        from rdflib import URIRef
        from rdflib import RDFS
        from rdflib import Literal
        #g = Graph()
        u = URIRef(u'http://example.com/foo')
        #g.add([u, RDFS.label, Literal('foo')])
        graph_in_memory.add((u, RDFS.label, Literal('foo')))
        graph_in_memory.commit()
        
        s=graph_in_memory.serialize(format='n3')
        
        print("graph size after add new triple: ", len(graph_in_memory))
        print(s)
if __name__ == '__main__':
    nIF2RDFProcessor= NIF2RDFProcessor()  
    nIF2RDFProcessor.test_load_rdf()  