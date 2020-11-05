# Reviewer Recommender-System  
The recommender algorithm works magic as we've all seen with the various recommendations that we get for products online in websites like flipkart,amazon, myntra and the main aspect of it is the accuracy of recommendation. So i wanted to know how it worked and achieve better accuracy and use it for educational purposes.   

With the increasing pressure on researchers to produce scientifically rigorous and relevant research, researchers need to find suitable publication outlets with the highest value and visibility for their manuscripts. Traditional approaches for discovering publication outlets mainly focus on manually matching research relevance in terms of keywords as well as comparing journal qualities, but other research relevant information such as social connections, publication rewards, and productivity of authors are largely ignored. To assist in identifying effective publication outlets i.e. reviewers and to support effective reviewer recommendations for manuscripts, a three dimensional profile-boosted research analytics frame work (RAF) that holistically considers relevance, connectivity, and productivity is proposed.  

### Key Definitions:  
- Relevance refers to exact allignment of topics covered in paper.  
- Quality refers to rewards by publications  
- Connectivity -widely accepted reviewer  
- Productivity-quality of reviewer(judged by their expertise,h index,reputation)  
- The h-index is defined as the maximum value of h such that the given author/ journal has published h papers that have each been cited at least h times.The h-index is an author-level metric that attempts to measure both the productivity and citation impact of the publications of a scientist or scholar.  

### Why is there the need of the Recommender System?  
Reviewers evaluate article submissions to journals based on the requirements of that journal, predefined criteria, and the quality, completeness and accuracy of the research presented. They provide feedback on the paper, suggest improvements and make a recommendation to the editor about whether to accept, reject or request changes to the article.  

### Holistic Approch towards evaluation of a perfect reviewer:  
Three different key performance indicators (KPIs),namely
> relevance index  
> productivity index  
> connectivity index  

are constructed to measure the strength of each dimension of the Recommender System. The relevance index calculates the research relatedness of a manuscript to a journal in terms of discipline and keywords. The productivity index measures the quality, quantity, citations, and impact of the author’s research. The connectivity index determines the popularity of the journals (i.e., widely used journals by similar experts) to be recommended. Finally, a unique matching algorithm based on the three KPIs is developed to achieve optimal assignment of journals to manuscripts.  

### Datasets Used:-
    • AMiner is a novel online academic search and mining system, and it aims to provide a systematic modeling approach to help researchers and scientists gain a deeper understanding of the large and heterogeneous networks formed by authors, papers, conferences, journals and organizations. The system is subsequently able to extract researchers' profiles automatically from the Web and integrates them with published papers by a way of a process that first performs name disambiguation.  
    • Neural Information Processing Systems (NIPS) is one of the top machine learning conferences in the world. It covers topics ranging from deep learning and computer vision to cognitive science and reinforcement learning.This dataset includes the title, authors, abstracts, and extracted text for all NIPS papers to date (ranging from the first 1987 conference to the current 2016 conference)  
    
The papers published in the period of 10 years from 1996-2005 were taken from AMiner Dataset. A dataframe was made with relevant fields.     
The data of 364 reviewers was taken from NIPS dataset.  

### Synopsis & Detailed Process:
    1. Profiling of Journals and Manuscripts
    • integrating three different information sources: subjective, objective, and social information. 
    • Specifically, subjective information refers to the information that is found in a manuscript or a journal (e.g., keywords presented in a manuscript’s ,keyword section). 
    • The information that is derivable from the title and abstract of a manuscript is considered objective information.
    • What the authors who publish articles in a journal and what their peers say about a journal is referred to as the social information of a journal.  

#### Processes:
-  The abstract of papers were extracted along with the publication year and authors. The relevance would also depend on the timeline of publication. The most recent published paper would be more relevant in comparision to the one published 10 years ago.  
- LDA model was trained on the whole paper and LDA vector was generated for every paper.  
- Before Applying LDA following operations are performed on the data.  

> #### Data Pre-processing  
> We will perform the following steps:  
>    • Tokenization: Split the text into sentences and the sentences into words. Lowercase the words and remove punctuation.  
>    • Words that have fewer than 3 characters are removed.  
>    • All stopwords are removed.  
>    • Words are lemmatized— words in third person are changed to first person and verbs in past and future tenses are changed into present.  
>    • Words are stemmed— words are reduced to their root form.  

_Topic modeling is a type of statistical modeling for discovering the “topics” that occur in a collection of documents.Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document to a particular topic._  
- The LDA vector of all the papers of a researcher published bw 1996-2005 were subjected to log function so that the most recent publication have higher weightage than the older ones.It has various parameters. The no of topics extracted were 50.

- The relevance index determines the matching degree between both researcher and reviewer profiles.  
- cosine similarity bw researcher and reviewer gives the relevance degree of the subjective key.(Cij)
-  RAKE(Rapid automatic keyword extraction) is used to extract keywords from titles and abstracts and the keyword section.It is a domain independent keyword     extraction algo which tries to determine key phrases in a body of text by analysing the frequency of word appearance and its co-occurence with other words         in text.
-  Jaccard similarity was used measure to generate the matching degree of both profiles.(Kij)
-  Rij=x(Cij)+(1-x)Kij; 0<x<1
-  To identify the best parameter value for α, we selected 30 published journal articles and their corresponding reviewers. We varied the value of α from 0.1 to 1 by fixing ρ at 0.4, and for each x value the accuracy (i.e., precision) of the top 3 predicated journals were computed. The value of x that gave best accuracy was selected.  

       
       

       
       2. Connectivity index:Identifying Widely Accepted Journals
       • To identify widely accepted journals by similar researchers, a collaboration network analysis was performed   
       and the connectivity index was calculated. The connectivity index measures the strength of the connection between   
       researchers, and it is used to identify potential journals in which similar researchers (e.g., co-authors) have published.
       • A node in the collaboration network represents a researcher. An edge between two nodes is constructed when one  
       researcher has co-authored with the other researcher. We first assign the weight w ij for a pair of vertices, which is defined as   
       the frequency of collaboration between two researchers. High weight implies more connectivity between the two researchers. We first use  
       a graph clustering method to identify groups of similar researchers (i.e., communities) in the collaboration network.  
       Hierarchical clustering, which is a traditional method for detecting community structure, is followed to derive an optimal community structure.
       • Graph clustering by Louvain was implemented to detect communities. Networkx best partition was used.
       • Communities are groups of nodes within a network that are more densely connected to one another than to other nodes.  
       Modularity is a metric that quantifies the quality . Louvain uses heirarchichal clustering  approach that maximises modularity to identify optimal   partitioning of nodes.
       Q k = ∑ i = 1 ( w II − a I2 ); Wll-same community; aI2-diff community
       Cij=0.05 if they are in diff community else 1.  
      

- Reviewer topic network(topic similarity as weight bw the nodes)  
- Reviewer paper network(frequency of co authored as weight  bw the nodes)  

      3.Productivity Index
      Da  =Publication quality; 
      Journal types=A,B,C
      qa=no of publications in journal of type A
      similarly qb,qc;
      D a = w A q A + w B q B + w C q C
      Ea= y(Ha)+(1-y)Da;
      Ea=Productivity Index; Ha=h-index  
    
- We consider the level of expertise of the author/authors before recommending the most productive publication outlet for their manuscript.We further assume that professionalism is reflected in the quality of research output and a patient researcher who has established a reputation can submit their manuscript to top-level journals and if unsuccessful they can submit it to the journal with lower rewards but with a high acceptance rate.  

      4. Recommending Journals for Manuscripts
- The key objective of our approach is to recommend
journals which maximize the relevance, that is, the match-
ing degree between the contents of a manuscript and the
contents of the publications of a journal and maximize the
connectivity index. Because the quality of the recommen-
dation largely depends on the quality of the publication
outlets, and their usefulness or appropriateness to the
author, there is a need to maintain a balance between an
author’s productivity and journal quality.
we recommend journals
with lower rewards but with a high acceptance rate. Thus,
we recommend all journals whose quality exceeds the pro-
ductivity of the authors

    ### Reference:
    - A Profile-Boosted Research Analytics Framework to Recommend Journals for Manuscripts  
    -Thushari Silva and Jian Ma, Chen Yang and Haidan Liang
    - A Robust Model for Paper-Reviewer Assignment  
    -Xiang Liu, Torsten Suel, Nasir Memon
