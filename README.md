# Offensive_Comment_Bias_Reduction

## Created By:

* __Kyle Hayes__

## Project Details:

Existing offensive speech detection models often learn to categorize discussion of marginalized identity groups with attacks against these groups themselves. As a result, these models retain significant biases against minority racial, ethnic, gender, and sexual identities. The purpose of this project is to create an offensive speech detection model for use on comment boards that minimizes false 'offensive' classifications by predicting not only whether a comment is offensive, but whether it relates to a specific identity group as well.

## Process Breakdown:

- **Business Understanding**:
  Internet comment boards have become an extremely popular way not only for people to communicate via social network, but for businesses to build relationships with their consumers. These businesses have incentive to find an automated model that remove offensive comments as quickly as possible. However, given the potential negative publicity that can surround machine learning systems that have accidentally internalized prejudice against identity groups, these businesses arguably have a greater incentive to insure that their machine learning systems minimize incorrect 'offensive' assignations.

- **Data Understanding**:
  I obtained my data from Kaggle's "Jigsaw Unintended Bias in Toxicity Classification" training dataset (https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data). It consists of 1.8 million Quora comments, which Google annotators have labelled according to their offensiveness. The resulting 'target' is the percentage of commentators who believed the comment to be offensive. In addition, roughly 405 thousand comments contained annotations regarding whether the comment specifically related to an identity group. For example, if a comment had the value '0.4' for 'female', it meant that 40% of annotators believed that the comment related to women as an identity. For my dataset, I used this subsection.
  In addition to these values, the data contains information about the comment sections, labels regarding specific types of offensiveness ('obscene', 'insult', etc.), as well as offensiveness labels obtained from a prior platform.

- **Data Preparation**:
  I will first remove the 1.4 million comments without identity annotations. I will binarize the 'target' variable by classifying any value equal to or greater than 0.5 as offensive, and I will binarize each 'identity' variable by classifying any value equal to or greater than 0.15 as relating to the identity. I will then create a second multinary 'target' variable that describes both whether the comment is offensive and whether it relates to at least one 'identity'. After separating the two target variables from the other features, I will then remove all features except for the comments themselves, and clean the text in preparation for vectorization.

- **Modeling**:
  To determine our baseline, I will vectorize the words in each comment, and perform a simple logistic regression using only the binary 'offensive' categorization as a target. For our final model, I will use a multinary target that reflects both offensiveness and the presence of 'identity group' content. In addition, this model will use a Keras deep learning algorithm.

- **Evaluation**:
  Since the inoffensive/offensive comment ratio is imbalanced, and since we have an interest in minimizing false 'offensive' assignations more than false 'inoffensive' assignations, I will use a custom cost function that penalizes/rewards each categorization appropriately. However, when searching for parameters, I will use the F1-score for the sake of simplicity to find the parameters that best balance our precision and recall. 

- **Deployment**:
  I will save and make available my final Keras model.

## Files in Repository:

* __README.md__ - A summary of the project.

* __technical_notebook.ipynb__ - Step-by-step walk through of the modeling/optimization process with rationale explaining decisions made. After cleaning and preparing the data, the notebook uses a gridsearch to create a logistic regression of the vectorized comments, and determines a baseline score using a custom cost function. For comparison, it also shows the score using only test data that contains 'identity' content. It then shows the process of creating a Keras deep-learning model, and compares the resulting scores (both for the entire test dataset and for only 'identity group' comments) to the prior model.

* __data_prep.py__ - Gathers and prepares data for analysis.

* __functions.py__ - Provides general functions used in the technical notebook, including Keras modeling and cost function calculation.

* __train_bias.csv__ - The raw dataset prior to data preparation.

* __target.csv__ - The cleaned target variables.

* __target.csv__ - The cleaned feature variables.

* __presentation.pptx__ - a presentation detailing the project and its results.