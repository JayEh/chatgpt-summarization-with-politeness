*Author's Comments:*
The setup for this project was to use ChatGPT API to synthesize a dataset of paragraphs, on various topics, and then to summarize those paragraphs. The nuance here is **not** to ask for a polite summarization, but to **ask politely** for a summarization and see how it impacts the result. The exact prompt is provided in the analysis below. I'm not sure if ChatGPT quite got that nuance about the test setup but the autonomous analysis capabilities remain impressive. All text below was written by ChatGPT, including the code and charts. I performed some collation on the results but did not alter the analysis provided by ChatGPT. I hope you enjoy the read and this demonstration inspires you to take the idea even further.


# Analyzing the differences in output between neutral and polite summarization requests with ChatGPT Code Interpreter [Alpha].
![234151926-8db2eae0-d85a-4e9a-adbd-2660265e8c0b](https://user-images.githubusercontent.com/20936780/236697702-09ab34e2-3f0e-46a0-a92c-8562a8f8fe58.png)

## Overview

This research project aimed to investigate the relationship between the length of input prompts and the length of generated summaries, as well as the impact of incorporating politeness on summarization length and quality. The study analyzed a dataset of Long Prompts, Neutral Responses, and Polite Responses, and employed various data analysis techniques to explore the factors that influence summary length, assess the impact of politeness on the summarization process, and evaluate the quality of the generated summaries in terms of similarity to the original prompts, sentiment consistency, and semantic content preservation.

## Project Description

The project consisted of several stages, including data collection, data analysis, and interpretation of the results. The dataset used in this study comprised Long Prompts, Neutral Responses, and Polite Responses, covering a diverse range of topics and text lengths. The data analysis involved multiple techniques, such as regression analysis, sentiment analysis, length ratio calculations, cosine similarity calculations, and t-SNE clustering, to provide a comprehensive understanding of the impact of politeness on summarization length and quality.

## Success and Implications

The study successfully identified a positive linear relationship between the length of input prompts and the length of generated summaries, with longer input prompts generally resulting in longer summaries. The addition of politeness was found to be associated with longer summaries compared to neutral summarizations, and this relationship was statistically significant. The analysis also revealed that both Neutral and Polite Responses exhibit a high degree of similarity to the original Long Prompts in terms of embeddings, suggesting that the summarization process preserves the semantic content of the original text.

The findings of this research have important implications for the development of natural language summarization systems. By understanding the factors that influence summary length and the impact of politeness on summarization length and quality, developers can design systems that generate summaries that are both informative and appropriate for the target audience and context. Furthermore, the insights gained from this study can contribute to the ongoing research in the field of natural language processing and help improve the performance of summarization systems in various applications, such as news summarization, document summarization, and content curation.

# The Impact of Politeness on Summarization Length and Quality

## Abstract
This study investigates the relationship between the length of input prompts and the length of generated summaries, as well as the impact of incorporating politeness on summarization length and quality. The analysis is based on a dataset of Long Prompts, Neutral Responses, and Polite Responses. The results indicate that longer input prompts generally result in longer summaries, and polite summarizations tend to be longer than neutral summarizations. Furthermore, the addition of politeness is associated with longer summaries compared to neutral summarizations, and this relationship is statistically significant. The study also explores the similarity between the generated summaries and the original prompts, the sentiment consistency, and the semantic content preservation. The findings provide valuable insights into the impact of politeness on the summarization process and can inform the development of natural language summarization systems.

## 1. Introduction
Automatic text summarization is an essential task in natural language processing (NLP) that aims to generate concise and informative summaries of longer texts. With the increasing amount of textual data available, the demand for efficient and accurate summarization systems has grown significantly. One aspect of summarization that has received less attention is the incorporation of politeness in the generated summaries. Politeness can play a crucial role in enhancing the readability and acceptability of summaries, especially in contexts where the target audience values politeness or when the content deals with sensitive topics.

This study investigates the relationship between the length of input prompts and the length of generated summaries, as well as the impact of incorporating politeness on summarization length and quality. The analysis is based on a dataset of Long Prompts, Neutral Responses, and Polite Responses, generated using two sets of summarization instructions: Neutral Instructions and Polite Instructions.

Neutral Instructions:
```
Summarize the following document/statement, between triple backticks (```), as shortly as possible. 
Ensuring that the summary:
1. Captures the main ideas and key points of the original text.
2. Is concise and significantly shorter than the original text, without losing essential information.
3. Maintains the original meaning and accurately represents the content.
4. Is written in clear, coherent, and grammatically correct language.
5. Avoids including any opinions, interpretations, or additional information not found in the original text.
6. Preserves the tone (neutral, positive, or negative) of the original text, if applicable.
<DOCUMENT_TOKEN>
```

Polite Instructions:
```
I would be truly grateful if you could kindly help me with summarizing the following document/statement, between triple backticks (```), as shortly as possible. 
While doing so, I kindly request you to consider the following aspects to ensure an accurate and meaningful summary:
1. Please make sure to capture the main ideas and key points of the original text.
2. It would be wonderful if the summary could be concise and significantly shorter than the original text while retaining essential information.
3. I appreciate your efforts in maintaining the original meaning and accurately representing the content.
4. If possible, please use clear, coherent, and grammatically correct language in the summary.
5. It would be great if you could refrain from including any opinions, interpretations, or additional information not found in the original text.
6. Lastly, kindly preserve the tone (neutral, positive, or negative) of the original text, if applicable.
<DOCUMENT_TOKEN>
Thank you so much for your assistance, and I truly value your expertise in helping me with this task.
```

The primary objectives of this research are to understand the factors that influence summary length, assess the impact of politeness on the summarization process, and evaluate the quality of the generated summaries in terms of similarity to the original prompts, sentiment consistency, and semantic content preservation.

## 2. Methodology

### 2.1 Dataset

The dataset used in this study consists of Long Prompts, Neutral Responses, and Polite Responses. Long Prompts are the original texts that need to be summarized, while Neutral Responses and Polite Responses are the generated summaries without and with politeness, respectively. The dataset includes a diverse range of topics and text lengths to ensure a comprehensive analysis.

### 2.2 Data Analysis

The data analysis consists of several steps to investigate the relationship between the length of input prompts and the length of generated summaries, as well as the impact of incorporating politeness on summarization length and quality.

### 2.2.1 Regression of Lengths: Long Prompts vs Summaries

A regression analysis is performed to examine the relationship between the length of Long Prompts and the lengths of Neutral and Polite Responses. Scatter plots and regression lines are used to visualize the data and identify trends.

### 2.2.2 Average Length Differences Between Responses

The average length differences between various pairs of responses (Long Prompt vs. Neutral Response, Long Prompt vs. Polite Response, and Neutral Response vs. Polite Response) are calculated and visualized using bar charts.

### 2.2.3 Average Neutral and Polite Distances

The average distances between the embeddings of Long Prompts and the embeddings of corresponding Neutral and Polite Responses are calculated and visualized using bar charts.

### 2.2.4 Sentiment Analysis

Sentiment scores of Long Prompts, Neutral Responses, and Polite Responses are calculated using the TextBlob library. Scatter plots are used to visualize the sentiment scores and identify trends.

### 2.2.5 Length Ratios

The average length ratios of summarized prompts (Neutral Response and Polite Response) to Long Prompts are calculated and visualized using bar charts.

### 2.2.6 Cosine Similarities Between Embeddings

The average cosine similarities between the embeddings of different pairs (Long Prompt vs. Neutral Response, Long Prompt vs. Polite Response, and Neutral Response vs. Polite Response) are calculated and visualized using bar charts.

### 2.2.7 t-SNE Clusters of Prompt, Neutral, and Polite Embeddings

t-distributed Stochastic Neighbor Embedding (t-SNE) is applied to the embeddings of Long Prompts, Neutral Responses, and Polite Responses to visualize the relationships between embeddings in a 2D plot.

## 3. Results

### 3.1 Regression of Lengths: Long Prompts vs Summaries
```python
# Import required libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.stats import ttest_rel
from textblob import TextBlob

# Load the JSON file
file_path = 'summarization_results.json'
with open(file_path, 'r') as file:
    summarization_results = json.load(file)

# Initialize lists to store lengths of texts
long_prompt_lengths = []
neutral_response_lengths = []
polite_response_lengths = []

# Calculate lengths of texts
for short_prompt, data in summarization_results.items():
    long_prompt = data['long_prompt']
    neutral_response = data['neutral_response']
    polite_response = data['polite_response']
    
    # Store lengths in the lists
    long_prompt_lengths.append(len(long_prompt))
    neutral_response_lengths.append(len(neutral_response))
    polite_response_lengths.append(len(polite_response))

# Create DataFrame for regression
df = pd.DataFrame({
    'Long_Prompt_Length': long_prompt_lengths,
    'Neutral_Response_Length': neutral_response_lengths,
    'Polite_Response_Length': polite_response_lengths
})

# Perform the paired t-test
t_statistic, p_value = ttest_rel(neutral_response_lengths, polite_response_lengths)

# Display the results
t_statistic, p_value
# Linear regression for Neutral Response
reg_neutral = LinearRegression().fit(np.array(long_prompt_lengths).reshape(-1, 1), neutral_response_lengths)
neutral_predicted = reg_neutral.predict(np.array(long_prompt_lengths).reshape(-1, 1))

# Linear regression for Polite Response
reg_polite = LinearRegression().fit(np.array(long_prompt_lengths).reshape(-1, 1), polite_response_lengths)
polite_predicted = reg_polite.predict(np.array(long_prompt_lengths).reshape(-1, 1))

# Plot the regression results
plt.figure(figsize=(12, 8))
sns.regplot(x='Long_Prompt_Length', y='Neutral_Response_Length', data=df, scatter_kws={'s': 10, 'color': 'blue'}, line_kws={'color': 'blue'}, label='Neutral Response')
sns.regplot(x='Long_Prompt_Length', y='Polite_Response_Length', data=df, scatter_kws={'s': 10, 'color': 'orange'}, line_kws={'color': 'orange'}, label='Polite Response')
plt.xlabel('Length of Long Prompt')
plt.ylabel('Length of Summaries')
plt.title(f'Regression of Lengths: Long Prompts vs Summaries\nPaired t-test p-value: {p_value:.2e}')
plt.legend()
plt.show()

```
![image](https://user-images.githubusercontent.com/20936780/236698582-8688dbd5-972d-4538-bce8-e68bf6a1d1ce.png)

The regression analysis revealed a positive linear relationship between the length of Long Prompts and the lengths of summaries, with longer input prompts generally resulting in longer summaries. The slopes of the regression lines for both Neutral and Polite Responses were similar, suggesting a consistent relationship between the length of Long Prompts and the lengths of summaries, regardless of whether the summary is neutral or polite. The paired t-test yielded a statistically significant p-value, indicating that the difference in mean summarization lengths between neutral and polite summarizations is statistically significant, with polite summarizations being longer than neutral summarizations.

### 3.2 Average Length Differences Between Responses
```python
# Initialize variables to store length differences
length_diff_long_neutral = []
length_diff_long_polite = []
length_diff_neutral_polite = []

# Initialize variables to store distances
neutral_distances = []
polite_distances = []

# Iterate over each item in the dictionary
for short_prompt, data in summarization_results.items():
    # Extract information from the inner dictionary
    long_prompt = data['long_prompt']
    neutral_response = data['neutral_response']
    polite_response = data['polite_response']
    neutral_distance = data['neutral_distance']
    polite_distance = data['polite_distance']
    
    # Calculate length differences
    length_diff_long_neutral.append(len(long_prompt) - len(neutral_response))
    length_diff_long_polite.append(len(long_prompt) - len(polite_response))
    length_diff_neutral_polite.append(len(neutral_response) - len(polite_response))
    
    # Store distances
    neutral_distances.append(neutral_distance)
    polite_distances.append(polite_distance)

# Calculate average length differences
avg_length_diff_long_neutral = np.mean(length_diff_long_neutral)
avg_length_diff_long_polite = np.mean(length_diff_long_polite)
avg_length_diff_neutral_polite = np.mean(length_diff_neutral_polite)

(avg_length_diff_long_neutral, avg_length_diff_long_polite, avg_length_diff_neutral_polite)

# Create bar chart for length differences
plt.figure(figsize=(8, 6))
plt.bar(
    ['Long-Neutral', 'Long-Polite', 'Neutral-Polite'],
    [avg_length_diff_long_neutral, avg_length_diff_long_polite, avg_length_diff_neutral_polite],
    color=['blue', 'orange', 'green']
)
plt.ylabel('Average Length Difference')
plt.title('Average Length Differences Between Responses')
plt.show()
```

![image](https://user-images.githubusercontent.com/20936780/236698061-92d609d2-563d-4114-ba76-9783557e7ca6.png)

The analysis of average length differences showed that both Neutral and Polite Responses are generally shorter than the original Long Prompts, which aligns with the goal of generating concise summaries. However, Polite Responses tend to be longer than Neutral Responses, as the addition of politeness results in longer summaries. The difference in length between Neutral and Polite Responses is statistically significant, as confirmed by the paired t-test conducted earlier.

### 3.3 Average Neutral and Polite Distances
```python
# Calculate average distances
avg_neutral_distance = np.mean(neutral_distances)
avg_polite_distance = np.mean(polite_distances)

# Create bar chart for distances
plt.figure(figsize=(8, 6))
plt.bar(
    ['Neutral Distance', 'Polite Distance'],
    [avg_neutral_distance, avg_polite_distance],
    color=['blue', 'orange']
)
plt.ylabel('Average Distance')
plt.title('Average Neutral and Polite Distances')
plt.show()

(avg_neutral_distance, avg_polite_distance)
```

![image](https://user-images.githubusercontent.com/20936780/236698074-dfb07275-67d9-4566-9c3d-91ce55270429.png)

The analysis of average distances revealed that Polite Responses tend to have a closer embedding distance to the Long Prompts compared to Neutral Responses. This suggests that, on average, Polite Responses are more similar to the original Long Prompts in terms of embedding representations compared to Neutral Responses.

### 3.4 Sentiment Analysis
```python
# Perform sentiment analysis on long prompts and summaries
for short_prompt, data in summarization_results.items():
    long_prompt = data['long_prompt']
    neutral_response = data['neutral_response']
    polite_response = data['polite_response']
    
    # Get sentiment scores
    long_prompt_sentiment = TextBlob(long_prompt).sentiment.polarity
    neutral_response_sentiment = TextBlob(neutral_response).sentiment.polarity
    polite_response_sentiment = TextBlob(polite_response).sentiment.polarity
    
    # Store sentiment scores in the lists
    long_prompt_sentiments.append(long_prompt_sentiment)
    neutral_response_sentiments.append(neutral_response_sentiment)
    polite_response_sentiments.append(polite_response_sentiment)
    
# Create a scatter plot of the sentiment scores
plt.figure(figsize=(10, 8))
plt.scatter(range(len(long_prompt_sentiments)), long_prompt_sentiments, c='blue', label='Long Prompt')
plt.scatter(range(len(neutral_response_sentiments)), neutral_response_sentiments, c='green', label='Neutral Response')
plt.scatter(range(len(polite_response_sentiments)), polite_response_sentiments, c='orange', label='Polite Response')
plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add line for neutral sentiment
plt.xlabel('Index')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Scores of Long Prompts and Summaries')
plt.legend()
plt.show()

# Calculate the average sentiment scores for long prompts, neutral responses, and polite responses
avg_sentiment_long_prompt = np.mean(long_prompt_sentiments)
avg_sentiment_neutral_response = np.mean(neutral_response_sentiments)
avg_sentiment_polite_response = np.mean(polite_response_sentiments)

(avg_sentiment_long_prompt, avg_sentiment_neutral_response, avg_sentiment_polite_response)
```

![image](https://user-images.githubusercontent.com/20936780/236698290-4b218dbb-838c-434a-9a10-d78d143b431e.png)

The sentiment analysis indicated that Long Prompts, Neutral Responses, and Polite Responses exhibit varying sentiment scores, with a slight tendency towards positive sentiment. The sentiment scores of the summaries are relatively consistent with the sentiment of the original Long Prompts. However, there are instances where the sentiment of the summary deviates from that of the original text.

### 3.5 Length Ratios
```python
# Initialize lists to store length ratios
length_ratio_neutral = []
length_ratio_polite = []

# Calculate length ratios
for short_prompt, data in summarization_results.items():
    long_prompt = data['long_prompt']
    neutral_response = data['neutral_response']
    polite_response = data['polite_response']
    
    # Compute length ratios
    ratio_neutral = len(neutral_response) / len(long_prompt) if len(long_prompt) > 0 else 0
    ratio_polite = len(polite_response) / len(long_prompt) if len(long_prompt) > 0 else 0
    
    # Store the computed length ratios in the lists
    length_ratio_neutral.append(ratio_neutral)
    length_ratio_polite.append(ratio_polite)

# Calculate average length ratios
avg_length_ratio_neutral = np.mean(length_ratio_neutral)
avg_length_ratio_polite = np.mean(length_ratio_polite)

# Create bar chart for length ratios
plt.figure(figsize=(8, 6))
plt.bar(
    ['Neutral Response', 'Polite Response'],
    [avg_length_ratio_neutral, avg_length_ratio_polite],
    color=['green', 'orange']
)
plt.ylabel('Average Length Ratio')
plt.ylim(0.4, 0.7)  # Rescale the y-axis to focus on the differences
plt.title('Average Length Ratios of Summarized Prompts to Long Prompts')
plt.show()

(avg_length_ratio_neutral, avg_length_ratio_polite)
```

![image](https://user-images.githubusercontent.com/20936780/236698390-a2128211-b7c8-454a-b743-354ceea671e6.png)

The analysis of length ratios showed that both Neutral and Polite Responses effectively condense the original Long Prompts, with Neutral Responses being more concise on average. The Polite Responses, while still effectively summarizing the content, tend to be longer due to the incorporation of polite language.

### 3.6 Cosine Similarities Between Embeddings
```python
# Initialize lists to store cosine similarities
cosine_similarity_prompt_neutral = []
cosine_similarity_prompt_polite = []
cosine_similarity_neutral_polite = []

# Calculate cosine similarities
for long_prompt, data in summarization_results.items():
    prompt_embedding = np.array(data['prompt_embedding']).reshape(1, -1)
    neutral_embedding = np.array(data['neutral_embedding']).reshape(1, -1)
    polite_embedding = np.array(data['polite_embedding']).reshape(1, -1)
    
    # Compute cosine similarities
    similarity_prompt_neutral = cosine_similarity(prompt_embedding, neutral_embedding)[0][0]
    similarity_prompt_polite = cosine_similarity(prompt_embedding, polite_embedding)[0][0]
    similarity_neutral_polite = cosine_similarity(neutral_embedding, polite_embedding)[0][0]
    
    # Store the computed similarities in the lists
    cosine_similarity_prompt_neutral.append(similarity_prompt_neutral)
    cosine_similarity_prompt_polite.append(similarity_prompt_polite)
    cosine_similarity_neutral_polite.append(similarity_neutral_polite)

# Calculate average cosine similarities
avg_cosine_similarity_prompt_neutral = np.mean(cosine_similarity_prompt_neutral)
avg_cosine_similarity_prompt_polite = np.mean(cosine_similarity_prompt_polite)
avg_cosine_similarity_neutral_polite = np.mean(cosine_similarity_neutral_polite)

# Create bar chart for cosine similarities with a rescaled y-axis
plt.figure(figsize=(8, 6))
plt.bar(
    ['Prompt-Neutral', 'Prompt-Polite', 'Neutral-Polite'],
    [avg_cosine_similarity_prompt_neutral, avg_cosine_similarity_prompt_polite, avg_cosine_similarity_neutral_polite],
    color=['blue', 'orange', 'green']
)
plt.ylim(0.95, 1.0)  # Rescale the y-axis to focus on the differences
plt.ylabel('Average Cosine Similarity')
plt.title('Average Cosine Similarities Between Embeddings (Rescaled)')
plt.show()

# Print useful information for analysis
(avg_cosine_similarity_prompt_neutral, avg_cosine_similarity_prompt_polite, avg_cosine_similarity_neutral_polite)
```

![image](https://user-images.githubusercontent.com/20936780/236698420-cb86573e-745f-4212-99fe-bf776a0bb677.png)

The analysis of cosine similarities indicated that both Neutral and Polite Responses exhibit a high degree of similarity to the original Long Prompts in terms of embeddings, suggesting that the summarization process preserves the semantic content of the original text. Additionally, Neutral and Polite Responses are highly similar to each other, indicating that the addition of politeness does not significantly alter the semantic content of the summaries.

### 3.7 t-SNE Clusters of Prompt, Neutral, and Polite Embeddings
```python
# Initialize lists to store embeddings
prompt_embeddings = []
neutral_embeddings = []
polite_embeddings = []

# Extract embeddings from the dictionary
for long_prompt, data in summarization_results.items():
    prompt_embedding = data['prompt_embedding']
    neutral_embedding = data['neutral_embedding']
    polite_embedding = data['polite_embedding']
    
    prompt_embeddings.append(prompt_embedding)
    neutral_embeddings.append(neutral_embedding)
    polite_embeddings.append(polite_embedding)

# Combine all embeddings into one array
all_embeddings = np.concatenate((prompt_embeddings, neutral_embeddings, polite_embeddings), axis=0)

# Apply t-SNE to the embeddings
tsne = TSNE(n_components=2, random_state=0)
embeddings_2d = tsne.fit_transform(all_embeddings)

# Split the transformed embeddings into separate arrays for plotting
prompt_embeddings_2d = embeddings_2d[:len(prompt_embeddings)]
neutral_embeddings_2d = embeddings_2d[len(prompt_embeddings):len(prompt_embeddings) + len(neutral_embeddings)]
polite_embeddings_2d = embeddings_2d[len(prompt_embeddings) + len(neutral_embeddings):]

# Create scatter plot of t-SNE clusters
plt.figure(figsize=(10, 8))
plt.scatter(prompt_embeddings_2d[:, 0], prompt_embeddings_2d[:, 1], c='blue', label='Prompt Embeddings')
plt.scatter(neutral_embeddings_2d[:, 0], neutral_embeddings_2d[:, 1], c='green', label='Neutral Embeddings')
plt.scatter(polite_embeddings_2d[:, 0], polite_embeddings_2d[:, 1], c='red', label='Polite Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Clusters of Prompt, Neutral, and Polite Embeddings')
plt.legend()
plt.show()

(prompt_embeddings_2d[:10], neutral_embeddings_2d[:10], polite_embeddings_2d[:10])
```

![image](https://user-images.githubusercontent.com/20936780/236698445-4f573e00-5788-44f6-8683-7464656bce3f.png)

The t-SNE plot showed distinct clusters for the embeddings of Long Prompts, Neutral Responses, and Polite Responses, suggesting that the embeddings of each category capture unique characteristics of the content. The proximity of Neutral and Polite Response clusters implies that the addition of politeness does not drastically alter the semantic content of the summaries.

## 4. Discussion

### 4.1 Length of Summaries

The results indicate that the length of input prompts is positively associated with the length of summaries. This information can be valuable for understanding the factors that influence summary length and for developing more efficient summarization systems.

### 4.2 Impact of Politeness on Summarization Length and Quality

The addition of politeness is associated with longer summaries compared to neutral summarizations. This relationship is statistically significant, as indicated by the low p-value from the paired t-test. The impact of politeness on summarization length and quality can be valuable for understanding the trade-offs between conciseness and politeness in the summarization process.

### 4.3 Similarity and Semantic Content Preservation

Both Neutral and Polite Responses exhibit a high degree of similarity to the original Long Prompts in terms of embeddings, suggesting that the summarization process preserves the semantic content of the original text. The addition of politeness does not significantly alter the semantic content of the summaries, as indicated by the high cosine similarities between Neutral and Polite Responses.

### 4.4 Sentiment Consistency

The sentiment scores of the summaries are relatively consistent with the sentiment of the original Long Prompts. However, there are instances where the sentiment of the summary deviates from that of the original text. This information can be valuable for understanding the emotional tone conveyed in the summaries and evaluating whether the sentiment of the original content is preserved in the summarization process.

### 4.5 Implications for Natural Language Summarization Systems

The findings of this study can inform the development of natural language summarization systems by providing insights into the impact of politeness on summarization length and quality. By understanding the trade-offs between conciseness and politeness, developers can design systems that generate summaries that are both informative and appropriate for the target audience and context.

## 5. Conclusion

This study investigated the relationship between the length of input prompts and the length of generated summaries, as well as the impact of incorporating politeness on summarization length and quality. The results indicate that longer input prompts generally result in longer summaries, and polite summarizations tend to be longer than neutral summarizations. Furthermore, the addition of politeness is associated with longer summaries compared to neutral summarizations, and this relationship is statistically significant. The study also explored the similarity between the generated summaries and the original prompts, the sentiment consistency, and the semantic content preservation. The findings provide valuable insights into the impact of politeness on the summarization process and can inform the development of natural language summarization systems.


### Examples of Summarization
#### Example 1.
Input Paragraph (len: 659)
```
Over the past decade, 3D printing technology has revolutionized the manufacturing industry with new possibilities. With 3D printing, manufacturers can create complex objects with intricate designs that would be impossible to produce using traditional manufacturing methods. This technology also allows for the creation of customized products and prototypes on a small scale, reducing waste and costs. The impact of 3D printing on manufacturing has been significant, with companies in various industries adopting the technology to improve their processes. From aerospace to healthcare, 3D printing has opened up new opportunities for innovation and efficiency.
```
Neutral Instructions Summarization (len: 370)
```
3D printing has revolutionized the manufacturing industry by allowing for the creation of complex objects with intricate designs, customized products, and prototypes on a small scale, reducing waste and costs. This technology has had a significant impact on various industries, from aerospace to healthcare, by opening up new opportunities for innovation and efficiency.
```
Polite Instructions Summarization (len: 585)
```
The past decade has seen a revolution in the manufacturing industry with the advent of 3D printing technology. This technology enables the creation of complex objects with intricate designs that are impossible to produce using traditional manufacturing methods. It also allows for the production of customized products and prototypes on a small scale, reducing waste and costs. 3D printing has had a significant impact on manufacturing, with companies in various industries adopting the technology to improve their processes and open up new opportunities for innovation and efficiency.
```

#### Example 2.
Input Paragraph (len: 793)
```
Television programming has undergone a significant transformation to cater to a diverse range of interests. Over the years, television networks have realized the importance of creating content that appeals to a variety of audiences. As a result, we now see an array of programs that cater to different age groups, ethnicities, and gender identities. From reality shows, dramas, and news programs to documentaries and sitcoms, there is something for everyone. This shift has not only made television an essential source of entertainment, but it has also opened up opportunities for underrepresented communities to see themselves reflected on screen. With the rise of streaming services and online platforms, this trend is likely to continue, providing a platform for diverse voices to be heard.
```
Neutral Instructions Summarization (len: 428)
```
Television programming has transformed to cater to diverse interests, with networks creating content for different age groups, ethnicities, and gender identities. This shift has made TV an essential source of entertainment and opened up opportunities for underrepresented communities to see themselves on screen. Streaming services and online platforms are likely to continue this trend, providing a platform for diverse voices.
```
Polite Instructions Summarization (len: 477)
```
Television programming has evolved to cater to a diverse range of interests, with networks creating content that appeals to different age groups, ethnicities, and gender identities. This shift has made television an essential source of entertainment and opened up opportunities for underrepresented communities to see themselves reflected on screen. Streaming services and online platforms are likely to continue this trend, providing a platform for diverse voices to be heard.
```

#### Example 3.
Input Paragraph (len: 837)
```
The integration of technology into the classroom has revolutionized the way educators teach and students learn. Technology has enabled educators to create interactive and engaging lessons that facilitate hands-on learning and cater to different learning styles. It has also made it possible for students to access educational resources beyond the classroom, such as online libraries and educational videos. Moreover, technology has facilitated communication between teachers and students, allowing for real-time feedback and personalized instruction. While some critics argue that technology can be a distraction and lead to a lack of face-to-face interaction, others believe that when used appropriately, technology can enhance the educational experience and prepare students for the digital world they will encounter outside of school.
```
Neutral Instructions Summarization (len: 354)
```
Technology has transformed education by enabling interactive and personalized learning, access to educational resources beyond the classroom, and real-time feedback between teachers and students. Critics argue that technology can be a distraction, but supporters believe it enhances the educational experience and prepares students for the digital world.
```
Polite Instructions Summarization (len: 445)
```
Technology has transformed the way teachers teach and students learn by enabling interactive and personalized instruction, facilitating access to educational resources beyond the classroom, and improving communication between teachers and students. While some argue that technology can be a distraction and hinder face-to-face interaction, others believe that it can enhance the educational experience and prepare students for the digital world.
```
