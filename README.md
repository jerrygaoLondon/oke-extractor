# oke-extractor
Concept Recognition and Interlink demo
========================

oke-challenge (https://github.com/anuzzolese/oke-challenge) Task 2: Class Induction and entity typing for Vocabulary and Knowledge Base enrichment.

(i) identify the type(s) of the given entity as they are expressed in the given definition, 
(ii) create a owl:Class statement for defining each of them as a new class in the target knowledge base, 
(iii) create a rdf:type statement between the given entity and the new created classes, and 
(iv) align the identified types, if a correct alignment is available, to a set of given types.

**Prerequisite to run:**<br/>
1) install python 3+<br/>
2) install libraries:<br/>

	$ sudo pip3 install flask
	$ sudo pip3 install -U nltk
	$ sudo pip3 install -U numpy scipy scikit-learn
	$ sudo pip3 install -U distance

	===Download NLTK data=====
	Run the Python interpreter and type the commands:
		import nltk
		nltk.download()
3) need the access to DBpedia

Test
========================
1. run test example<br/>
  - <pre><code><root>/oke-extractor/oke/oak/$ /usr/bin/python3.4 okeConceptRecogniser.py </code></pre>
  <br/>
  Directly run okeConceptRecogniser.py will process test file in "../test/task2_test.ttl" and output result in "../test/task2_test_output.ttl"

2. run with webservice<br/>
    **start webservice**<br/>
	- <pre><code><root>/oke-extractor/oke/oak/$/usr/bin/python3.4 conceptRecognitionWS.py</code></pre>
    <br/>
	This will start a web service via localhost:5000 with a RESTful API "http://localhost:5000/extract". API accepts N3 formatted NIF data input and output enriched N3 formatted NIF data. oke.oak.TestOKEConceptRecogniserWS.py gives an example how to test the API.
	
	Example test via cURL:
	<pre><code>curl -i -X POST http://localhost:5000/extract -H "Content-Type: text/xml" --data-binary "@path-to-file\task2_test.ttl"</code></pre>
	
	
  