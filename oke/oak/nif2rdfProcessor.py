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
        
    def load_rdf(self, filePath, _format='n3'):
        
        goldstandards_file = open(filePath, "r",encoding="utf8")
        
        graph_in_memory = rdflib.Graph("IOMemory")
        return graph_in_memory.parse(file=goldstandards_file,format=_format)
    
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
        query+='?taIdentRef a ?type.'
        query+='FILTER(?type != owl:Class)} order by ?beginIndex'
        
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
        query='select distinct ?anchorOf ?type ?dulClass ?beginIndex ?endIndex ?refContext '
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
                        str(relations[i]['refContext'])) for i in range(0, len(relations))]
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
            entityClasses.append(entityClass)
        
        entity.isInstOfEntityClasses=entityClasses
        currentContext.entity=entity
        return currentContext        
        
    def get_task_entities_labelledType(self,graph_in_memory):
        return None
        
        