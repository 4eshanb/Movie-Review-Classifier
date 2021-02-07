# Movie-Review-Classifier

Given the text of a movie review, multiple classifiers are used to know whether it is expressing a positive or
a negative opinion about a movie.

### Data  
There is a total of 500 movie reviews: 250 that have been given a positive rating (an overall score
\> 5 out of 10) and 250 that have been given a negative rating (an overall score < 5). 
These reviews havebeen split into training, development, and testing files. 

Each dataset contains a list of reviews. Each review has two facets, Overall, which specifies whether the
review is positive or negative, based on the score from 1-10 and Text, which gives the textual review. Each
review is separated by a period character (.)

### Binning 
First the raw counts of values of each
feature are used. Then the binned value of raw counts are used. For example, imagine the word the occurs 10 times, food
occurs 2 times, and the bigram the food occurs 1 time.
The string vectors the same as before except now using the raw counts.   
Hence the new feature vector from the previous example would now be:  
> UNI_the:10 UNI_food:2 BIGRAM_the_food:1

If you apply binning, with results in bins of 0, 1, 2, or 3, the new feature vector from the previous example
now becomes:  
> UNI_the:3 UNI_food:2 BIGRAM_the_food:1  


