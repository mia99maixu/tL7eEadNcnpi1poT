# Potential Talent (NLP)

### Background:

As a talent sourcing and management company, we are interested in finding talented individuals for sourcing these candidates for technology companies. Finding talented candidates is not easy, for several reasons. The first reason is one needs to understand what the role is very well to fill that spot, this requires understanding the client’s needs and what they are looking for in a potential candidate. The second reason is one needs to understand what makes a candidate shine for the role we are in search of. Third, where to find talented individuals is another challenge. The nature of our job requires a lot of human labor and is full of manual operations. Towards automating this process we want to build a better approach that could save us time and finally help us spot potential candidates that could fit the roles we are in search for. Moreover, going beyond that for a specific role we want to fill in we are interested in developing a machine learning powered pipeline that could spot talented individuals, and rank them based on their fitness.  We are right now semi-automatically sourcing a few candidates, therefore the sourcing part is not a concern at this time but we expect to first determine best matching candidates based on how fit these candidates are for a given role. We generally make these searches based on some keywords such as “full-stack software engineer”, “engineering manager” or “aspiring human resources” based on the role we are trying to fill in. These keywords might change, and you can expect that specific keywords will be provided. Assuming that we were able to list and rank fitting candidates, we then employ a review procedure, as each candidate needs to be reviewed and then determined how good a fit they are through manual inspection. This procedure is done manually and at the end of this manual review, we might choose not the first fitting candidate in the list but maybe the 7th candidate in the list. If that happens, we are interested in being able to re-rank the previous list based on this information. This supervisory signal is going to be supplied by starring the 7th candidate in the list. Starring one candidate actually sets this candidate as an ideal candidate for the given role. Then, we expect the list to be re-ranked each time a candidate is starred. 


### Data Description:  

Attributes:

id: unique identifier for candidate (numeric)

job_title: job title for candidate (text)

location: geographical location for candidate (text)

connections: number of connections the candidate has, 500+ means over 500 (text)

Output (desired target): fit - how fit the candidate is for the role? (numeric, probability between 0-1)
 
Keywords: “Aspiring human resources” or “seeking human resources”

### Goal(s):

Predict how fit the candidate is based on their available information (variable fit)

Rank candidates based on a fitness score.

Re-rank candidates when a candidate is starred.

### Methodology:

1.	Data Preparation:
- TF-IDF: Converts job_title text into numerical features.
- Label Encoding: Converts categorical location data into numerical labels.
- Group Assignment: Group candidates into a single query. (For larger datasets, you might want to group candidates by role or search query.)

3.	Ranking:
- Score calculate: Extract Initial Fitness Scores Based on Job Title Similarity.
- Ranked Fitness Scores: After getting the fitness scores, simply ranking them from the probabilities.
 

3.	Dynamic Re-ranking:
- Supervisory Signal: After a candidate is starred, the ranking is adjusted dynamically. This can be done by re-weighting features or manually boosting the score of the starred candidate.

#### Dataset after fitness scores calculated

|index|id|job\_title|location|connection|fit|
|---|---|---|---|---|---|
|0|1|2019 C\.T\. Bauer College of Business Graduate \(Magna Cum Laude\) and aspiring Human Resources professional|Houston, Texas|85|0\.32979136199300013|
|1|2|Native English Teacher at EPIK \(English Program in Korea\)|Kanada|500+ |0\.0|
|2|3|Aspiring Human Resources Professional|Raleigh-Durham, North Carolina Area|44|0\.9367369336368755|
|3|4|People Development Coordinator at Ryan|Denton, Texas|500+ |0\.0|
|4|5|Advisory Board Member at Celal Bayar University|İzmir, Türkiye|500+ |0\.0|

4. This approach focuses exclusively on the job_title feature to predict the fit score. It leverages text-based features and a regression model to rank candidates according to their job titles’ relevance to a specific role. This is a straightforward and interpretable approach that can be further enhanced by adding more advanced text-processing techniques or by incorporating more features if needed in the future.
5. We also tried the state-of-art similarity technique Word2Vec, Fastext and SentenceTransfomer.

### Conclusion:

We are interested in a robust algorithm, after starred one of the candidate the re-rank model come to calculate the similerity more relevan to the data point, so suggest to mannual select the top candidate as the model criteria

By applying a series of filters based on keywords, connections, location, and fit score, you can systematically remove unqualified or irrelevant candidates from the list. Set the condition such like location or connection over 500. Remove the candidates not meet the requirement. Train a model from historical data. The filters can be adjusted based on the specific requirements of the role, making this approach adaptable to different job searches.

Instead of manually setting a cut-off point, use historical data to learn what an effective cut-off point looks like for different roles. This approach can help minimize the chances of losing high-potential candidates.

Train the model multi time with slightly change of the criteria, or apply different technique for training




