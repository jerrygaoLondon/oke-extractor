'''
Created on 8 May 2015

@author: jieg
'''
import sys
sys.path.append("../../")

class DBpediaQuerier(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
    
    def dbpedia_query_rdftypes(self, entity_dbpedia_URI, endpoint="http://galaxy.dcs.shef.ac.uk:8893/sparql"):
        '''
        Linked Data Discovery for alignment by DBpedia Entity URI
        
        query rdf types in dbpedia dataset
        return rdf type URI set
        '''
        query="select ?type where { {<%s> a ?type . FILTER(?type != owl:Thing)} " % entity_dbpedia_URI
        query+= " union {<%s> dbpedia-owl:wikiPageRedirects ?entity . ?entity rdf:type ?type . FILTER(?type != owl:Thing)} " % entity_dbpedia_URI
        query+=" } "
        results = self.dbpedia_query(query)
        
        entity_rdftypes=set()
        for results in results["results"]["bindings"]:
            entity_rdftypes.add(results['type']['value'])
        return entity_rdftypes
    
    def dbpedia_query_dcterm_subject(self, entity_dbpedia_URI, endpoint="http://galaxy.dcs.shef.ac.uk:8893/sparql"):
        '''
        query rdf types in dbpedia dataset
        return rdf type URI list
        '''
        query="select ?subject_label where {<%s> dcterms:subject ?subject . ?subject rdfs:label ?subject_label .}" % entity_dbpedia_URI
        results = self.dbpedia_query(query)
        
        entity_rdftypes=set()
        for results in results["results"]["bindings"]:
            entity_rdftypes.add(results['subject_label']['value'])
        return entity_rdftypes
     
    def dbpedia_query(self,sparql_query,endpoint="http://galaxy.dcs.shef.ac.uk:8893/sparql"):
        '''
        return results in json format
        example:
        {'head': {'link': [], 'vars': ['type']}, 
         'results': {'distinct': False, 'ordered': True, 
             'bindings': [{'type': {'value': 'http://schema.org/Organization', 'type': 'uri'}}, 
                         {'type': {'value': 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#Agent', 'type': 'uri'}}, 
                         {'type': {'value': 'http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#SocialPerson', 'type': 'uri'}}, 
                         {'type': {'value': 'http://dbpedia.org/ontology/Agent', 'type': 'uri'}},
                         {'type': {'value': 'http://dbpedia.org/ontology/MilitaryUnit', 'type': 'uri'}}, 
                         {'type': {'value': 'http://dbpedia.org/ontology/Organisation', 'type':'uri'}}]}}
        '''
        from SPARQLWrapper import SPARQLWrapper, JSON
        sparql = SPARQLWrapper(endpoint)
        
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        return results
