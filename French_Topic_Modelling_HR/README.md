## Project Definition : Finding Most Important HR Skills for each French Cities
-> In this project, python >= 3.7.x is used.


### Project Setup
```language-bash
    pip3 install -r requirements.txt
```


### Read HR data 
-> Put Main Csv file to "csv file" folder. (congig.ini) 
-> Make proper adjustments from using config.ini
-> The Code below will be enough for running.

```language-bash
    python3.7 main.py
```


### Understanding the Work:
-> Reading the main csv file which includes City and Offer Information.
-> Cleaning the code by making all letters lowercase, cleaning non-letters, removing stopwords and too short words.
-> Counts every word's occurences based on city using Countvectorizer.
-> Traning Latent Dirichlet Algorithm (LDA) model with Counted values for each city.
-> LDA model trained to find 10 most important HR topics for each city.
-> After training LDA model, TSNE algorithm applied to reduce dimensioality in PCA manner.
-> After finding most important topics, showing cluster graphs for each cities topic with most important content with Bokeh library.


# Output
-> It gives 13 seperate html files for each city in data. The output file looks like : alsace-champagne-ardenne-lorraine.html, ile-de-france.html and so forth..
