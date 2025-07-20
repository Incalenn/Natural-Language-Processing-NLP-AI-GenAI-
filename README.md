# Sentiment Analysis on Allociné Movie Reviews

**Exam - Session 15 | DataScientest**  
This project is a practical application of text classification using Natural Language Processing (NLP) techniques on French-language movie reviews scraped from Allociné. The goal is to automatically classify movie reviews as positive or negative and explore text features using visualization, entity recognition, and similarity-based classification.

---

## Dataset

- **Source:** Allociné (French movie review website)  
- **Files:**
  - `allocine_train.csv` – training data (160,000 reviews)
  - `allocine_test.csv` – test data
- **Features:**
  - `review`: the raw text of the review
  - `polarity`: 0 = negative, 1 = positive

---

## Project Steps

### 1. Data Exploration

#### Load Dataset

```python
import pandas as pd

df = pd.read_csv("allocine_train.csv")
df = df[['review', 'polarity']]
df.head(10)
```

#### Polarity Distribution

```python
df['polarity'].value_counts()
```

**Output:**

```
1    80587
0    79413
```

#### Review Length by Polarity

```python
import matplotlib.pyplot as plt
import seaborn as sns

df['length'] = df['review'].apply(lambda x: len(str(x)))

plt.figure(figsize=(12, 6))
sns.boxplot(x='polarity', y='length', data=df)
plt.title("Review Length by Polarity")
plt.xlabel("Polarity")
plt.ylabel("Length")
plt.show()
```
**Output:**

![image1](image1.png)
---

### 2. Word Clouds

#### Text Cleaning Using spaCy

```python
import spacy
nlp = spacy.load("fr_core_news_lg")

def cleaning(text):
    doc = nlp(text)
    cleaned = [token.lemma_.lower() for token in doc
               if not token.is_stop and not token.is_punct and token.is_alpha]
    return " ".join(cleaned)

sample_df = df.sample(frac=0.01, random_state=42).copy()
sample_df['cleaned_review'] = sample_df['review'].apply(lambda x: cleaning(str(x)))
```

#### Word Clouds for Positive and Negative Reviews

```python
from wordcloud import WordCloud

text_pos = " ".join(sample_df[sample_df['polarity'] == 1]['cleaned_review'])
text_neg = " ".join(sample_df[sample_df['polarity'] == 0]['cleaned_review'])

wc_pos = WordCloud(background_color="white").generate(text_pos)
wc_neg = WordCloud(background_color="white").generate(text_neg)
```

#### Removing Common Words for Better Contrast

```python
from collections import Counter

words_pos = text_pos.split()
words_neg = text_neg.split()

count_pos = Counter(words_pos)
count_neg = Counter(words_neg)

common_words = set([word for word in count_pos if word in count_neg and count_pos[word] + count_neg[word] > 10])

text_pos_filtered = " ".join([w for w in words_pos if w not in common_words])
text_neg_filtered = " ".join([w for w in words_neg if w not in common_words])

wc_pos_filtered = WordCloud(background_color="white").generate(text_pos_filtered)
wc_neg_filtered = WordCloud(background_color="white").generate(text_neg_filtered)
```

---

### 3. Named Entity Recognition (NER)

#### Count Person Entities in Each Review

```python
def count_pers(text):
    doc = nlp(text)
    return len([ent for ent in doc.ents if ent.label_ == "PER"])

sample_df['pers'] = sample_df['review'].apply(lambda x: count_pers(str(x)))
```

#### Example Extracted Entities

```python
entities = []

for text in sample_df['review'].head(100):
    doc = nlp(text)
    entities.extend([ent.text for ent in doc.ents])
    if len(entities) >= 5:
        break

print(entities[:5])
```

**Output:**

```
['Paris', 'Référence', 'Bogart', 'Le Violent', 'Hitchcock']
```

#### Violin Plot: Person Entities by Polarity

```python
plt.figure(figsize=(10, 6))
sns.violinplot(x='polarity', y='pers', data=sample_df)
plt.title("Distribution of Named Person Entities by Polarity")
plt.xlabel("Polarity (0 = Negative, 1 = Positive)")
plt.ylabel("Count of Person Entities")
plt.show()
```

**Observation:** The number of named persons (`PER`) is not significantly different between positive and negative reviews.

---

### 4. Classification via Text Similarity (spaCy)

#### Reference Sentences

```python
doc_pos = nlp("Absolutely amazing! Beautiful direction and perfect acting. 1h50 of joy.")
doc_neg = nlp("Very slow and boring. I definitely do not recommend watching this film.")
```

#### Predicting Sentiment by Similarity

```python
from sklearn.metrics import accuracy_score

def predict_polarity(text):
    doc = nlp(text)
    sim_pos = doc.similarity(doc_pos)
    sim_neg = doc.similarity(doc_neg)
    return 1 if sim_pos > sim_neg else 0

sample_df['predicted_polarity'] = sample_df['review'].apply(predict_polarity)
accuracy = accuracy_score(sample_df['polarity'], sample_df['predicted_polarity'])
print(f"Accuracy with spaCy similarity: {accuracy:.3f}")
```

**Output:**

```
Accuracy with spaCy similarity: 0.612
```

#### Most Similar Positive Review

```python
similarities = sample_df['review'].apply(lambda x: nlp(x).similarity(doc_pos))
idx_most_similar = similarities.idxmax()

print("Most similar positive review:")
print(sample_df.loc[idx_most_similar, 'review'])
print(f"Similarity Score: {similarities[idx_most_similar]:.3f}")
```

**Example Output:**

```
"My little girls (10 and 6 years old) and I loved it. The drawings are beautiful as well as the music. The poetry of Nature and Life shown through the images, the simplicity of the drawings, the rhythm... made us feel really good."
Similarity Score: 0.861
```

---

## Conclusion

- The spaCy similarity method is quick and easy to implement but not sufficiently accurate.
- Advanced models like supervised learning or transformer-based approaches (e.g., CamemBERT) would likely improve performance significantly.

---

## Skills Applied

- French text preprocessing with spaCy  
- Data visualization using WordCloud and Seaborn  
- Named Entity Recognition (NER)  
- Text similarity-based sentiment classification  
- Model evaluation with accuracy score  

---

## Suggestions for Improvement

- Train traditional classifiers (Logistic Regression, Naive Bayes, etc.)
- Use TF-IDF or contextual embeddings (e.g., CamemBERT, Sentence Transformers)
- Build a full NLP pipeline: Preprocessing → Feature Extraction → Model Training → Evaluation

---

## Resources

- [spaCy – French NLP Models](https://spacy.io/usage/models/fr)  
- [WordCloud Python Library](https://github.com/amueller/word_cloud)  
- [Seaborn Violinplot Documentation](https://seaborn.pydata.org/generated/seaborn.violinplot.html)

---

## Instructions to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/allocine-sentiment-analysis.git
