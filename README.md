# glucose
Levels

Based on: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005143#references

Objective was to predict responses to foods using clinical + nutrition + CGM data 

File summary:
1. main.py runs everything (except unit tests)
2. data_loader.py reads .tsv and .db files and performs some data processing/cleaning 
3. eda.py performs exploratory data analysis 
4. model.py performs feature eng and has the logistic classifier that was use 
5. unittests.py has a couple sample unit tests 
