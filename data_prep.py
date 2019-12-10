"""
Module gathers and prepares data for analysis.
"""
import re
# pylint: disable=unused-import
import swifter
# pylint: enable=unused-import
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import functions as f


# pylint: disable=unnecessary-lambda
def annotated(data):
    """Drops non-annotated data."""

    data = data.loc[~(data.asian.isna())]
    data = data.loc[~(data.comment_text.isna())]
    data = data.reset_index(drop=True)
    return data


def target_grouping(row):
    """
    Creates new target incorporating identities.

    '0': no identities, inoffensive.
    '1': at least one identity, inoffensive.
    '2': no identities, offensive.
    '3': at least one identity, offensive.

    Parameters:
    row: individual datapoint.

    Returns:
    val (int): integer for use in updated target.
    """

    val = 0
    if row['groups'] == 0:
        if row['target'] == 0:
            val = 0
            return val
        val = 2
        return val
    if row['target'] == 0:
        val = 1
        return val
    val = 3
    return val


def identities(data):
    """
    Categorizes whether an identity is mentioned in each comment.

    Each 'identity_list' column refers a specific identity group, with each
    value representing the percentage of annotators believing the comment
    to refer to the identity group. Function categorizes values >= 0.15 as
    positive identification, and classifies comments based on whether any
    identity groups have been positively identified.

    Parameters:
    data: feature variable database.

    Returns:
    data: feature variable database with updated 'groups' feature.
    """

    identity_list = list((data.iloc[:, 8:32]).columns)
    for iden in identity_list:
        data[f'{iden}'] = data[iden].swifter.apply(lambda x: x if x >= 0.15
                                                   else 0)
    data['groups'] = data[identity_list].sum(axis=1)
    data['groups'] = data.groups.swifter.apply(lambda x: 0 if x == 0 else 1)
    data = data.drop(columns=identity_list)
    return data


def target_cols(data):
    """
    Categorizes whether target is offensive.

    'Target' column refers to percentage of annotators that found
    the comment at least mildly offensive. Function categorizes
    values >= 0.5 as offensive. Function also creates second target group
    describing both comment offensiveness and whether comment refers to
    an identity group.

    Parameters:
    data: feature variable database.

    Returns:
    data: feature variable database with updated 'offensive_and_identity'
          feature.
    """

    data['target'] = data.target.swifter.apply(lambda x: 1 if x >= .5 else 0)
    data['offensive_and_identity'] = data.swifter.apply(lambda row:
                                                        target_grouping(row),
                                                        axis=1)
    data = data.rename(columns={'target': 'offensive'})
    data = data.drop(columns=['groups'])
    return data


def column_change(data):
    """Removes unnecessary columns."""

    data = data[['comment_text']]
    return data


def comments_obscene(data):
    """
    Cleans words with punctuation obscuring obscenity.

    Parameters:
    data: feature variable database.

    Returns:
    data: feature variable database with updated 'comment_text' feature.
    """

    data['comment_text'] = data['comment_text'].swifter.apply(lambda x:
                                                              x.lower())
    data['comment_text'] = data['comment_text'].str.replace(r'a\*\*\*\*\*e',
                                                            'asshole')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'a\$\$hole', 'asshole'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (' ahole ', ' asshole '))
    data['comment_text'] = data['comment_text'].str.replace(r'a\*\*', 'ass')
    data['comment_text'] = data['comment_text'].str.replace(r'a\$\$', 'ass')
    data['comment_text'] = data['comment_text'].str.replace(r'sh\*t', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r's\*it', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'\*hit', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'\$hit', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'sh\*\*\*\*ss',
                                                            'shitless')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'bull \(sh\*\*\*er\)', 'bullshitter'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'b\*\*\*\*\*\*\*', 'bullshit'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'b\*\*\*\*\*\*t', 'bullshit'))
    data['comment_text'] = data['comment_text'].str.replace(r'sh\*\*\*er',
                                                            'shitter')
    data['comment_text'] = data['comment_text'].str.replace(r'sh\*\*t', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r' s\-\-t',
                                                            ' shit')
    data['comment_text'] = data['comment_text'].str.replace(r'sh\*\*', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r's\*\*t', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r's\*\*\*', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'b\*tch', 'bitch')
    data['comment_text'] = data['comment_text'].str.replace(r'bit\*h', 'bitch')
    data['comment_text'] = data['comment_text'].str.replace(r'f\*ck', 'fuck')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\*\*\*\*\*g', 'fucking'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\*\*\*\*d', 'fucked'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\*\*\*\*\*\*', 'fucking'))
    data['comment_text'] = data['comment_text'].str.replace(r'f\*\*\*', 'fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'f\-\-\-\-\-g',
                                                            'fucking')
    data['comment_text'] = data['comment_text'].str.replace(r'f\*\*k', 'fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'p\*ssy', 'pussy')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'pu\$\$y', 'pussy'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'pu\#\#y', 'pussy'))
    return data


def comments_obscene2(data):
    """
    Cleans words with punctuation obscuring obscenity.

    Parameters:
    data: feature variable database.

    Returns:
    data: feature variable database with updated 'comment_text' feature.
    """

    data['comment_text'] = (data['comment_text'].str.replace
                            (r'n\*gger', 'nigger'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r' n\*\*\*\*\*', ' nigger'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'nigg\*r', 'nigger'))
    data['comment_text'] = data['comment_text'].str.replace(r'f\*ggot',
                                                            'faggot')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\@gg0t', 'faggot'))
    data['comment_text'] = data['comment_text'].str.replace(r'f\@g', 'fag')
    data['comment_text'] = data['comment_text'].str.replace(r'h\*\*l', 'hell')
    data['comment_text'] = data['comment_text'].str.replace(r'h\*ll', 'hell')
    data['comment_text'] = data['comment_text'].str.replace(r'c\*nt', 'cunt')
    data['comment_text'] = data['comment_text'].str.replace(r'c\*unt', 'cunt')
    data['comment_text'] = data['comment_text'].str.replace(r'd\*\*\*\*\*\*',
                                                            'damned')
    data['comment_text'] = data['comment_text'].str.replace(r'sl\*t', 'slut')
    data['comment_text'] = data['comment_text'].str.replace(r'tw\*t', 'twat')
    data['comment_text'] = data['comment_text'].str.replace(r'wh\*re', 'whore')
    data['comment_text'] = data['comment_text'].str.replace(r'g\*y', 'gay')
    data['comment_text'] = data['comment_text'].str.replace(r'pr\*ck', 'prick')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'he\*l h\*tler', 'heil hitler'))
    data['comment_text'] = data['comment_text'].str.replace(r's\*x', 'sex')
    data['comment_text'] = data['comment_text'].str.replace(r'an\*l', 'anal')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'du\*b', 'dumb'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'm\*\*\*\*m', 'muslim'))
    data['comment_text'] = data['comment_text'].str.replace(r'mex\*\*\*\*',
                                                            'mexican')
    return data


def clean_punct(data):
    """
    Removes non-stopping punctuation, puts space between
    stopping punctuation.

    Parameters:
    data: feature variable database.

    Returns:
    data: feature variable database with updated 'comment_text' feature.
    """

    punct_list = [r'\-', r'\_', r"\'", r'\\', r'\/', r'\,', r'\*', r"\:",
                  r"\;", r"\(", r"\)", r"\{", r"\}", r"\[", r"\]", r"\|",
                  r"\<", r"\>", r"\#", r"\@", r'\%', r'\^', r'\+', r'\=',
                  r'\"', r'\&']
    for punct in punct_list:
        data['comment_text'] = data.comment_text.str.replace(f'{punct}', " ")
    return data


def punct_space(data):
    """
    Puts space between stopping punctuation and letters.

    Parameters:
    data: feature variable database

    Returns:
    data: feature variable database with updated 'comment_text' feature.
    """

    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                'w', 'x', 'y', 'z']
    punct_list = [r'\.', r'\?', r'\!']
    for punct in punct_list:
        for let in alphabet:
            data['comment_text'] = (data.comment_text.str.replace
                                    (f'{punct}{let}', f'{punct} {let}'))
    return data


def whitespace(data):
    """Reduces whitespace in comments."""

    i = 0
    while i <= 10:
        data['comment_text'] = data.comment_text.str.replace('  ', ' ')
        i += 1
    return data


def tokenize(data):
    """Tokenizes data."""

    word_tokenizer = RegexpTokenizer(r'\w+')
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: word_tokenizer.tokenize(x)))
    return data


def rejoin(data):
    """Rejoins tokenized data."""

    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: ' '.join(x)))
    return data


def replace_halfwords(data):
    """
    Replaces part words from tokenized words.

    Parameters:
    data: feature variable database

    Returns:
    data: feature variable database with updated 'comment_text' feature.
    """

    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('i m ', 'i am ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('you re ', 'you are ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('he s ', 'he is ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('she s ', 'she is ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('it s ', 'it is ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('we re ', 'we are ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('they re ', 'they are ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('don t ', 'do not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('didn t ', 'did not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('doesn t ', 'does not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('wasn t ', 'was not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('weren t ', 'were not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('won t ', 'will not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('hasn t ', 'has not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('hadn t ', 'had not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('wouldn t ', 'would not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('i ll ', 'i will ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('he ll ', 'he will ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('she ll ', 'she will ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('you ll ', 'you will ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('we ll ', 'we will ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('they ll ', 'they will ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('aren t ', 'are not ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('i ve ', 'i have ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('you ve ', 'you have ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('we ve ', 'we have ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('they ve ', 'they have ')))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.replace('can t ', 'cannot ')))
    return data


def replace_three(word_list):
    """
    Replaces letters that appear more than three times in a row.

    Parameters:
    word_list (list): list of words for datapoint

    Returns:
    new_list (list): updated list of words for datapoint.
    """

    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    new_list = []
    for word in word_list:
        new_word = str(pattern.sub(r"\1\1", word))
        new_list.append(new_word)
    return new_list


def misspelled(word_list):
    """
    Changes some misspelled words.

    Parameters:
    word_list (list): list of words for datapoint

    Returns:
    new_list (list): updated list of words for datapoint.
    """

    newlist = []
    for word in word_list:
        if word == 'bicthes':
            newlist.append('bitches')
        elif word == 'fukin':
            newlist.append('fucking')
        elif word == 'fectless':
            newlist.append('feckless')
        elif word == 'trustworhiness':
            newlist.append('trustworthiness')
        elif word == 'awfucl':
            newlist.append('awful')
        elif word == 'sextiimg':
            newlist.append('sexting')
        elif word == 'licemse':
            newlist.append('license')
        elif word == 'http':
            newlist.append('')
        elif word == 'ww':
            newlist.append('')
        else:
            newlist.append(word)
    return newlist


def misspelled2(word_list):
    """
    Changes some misspelled words.

    Parameters:
    word_list (list): list of words for datapoint

    Returns:
    new_list (list): updated list of words for datapoint.
    """

    newlist = []
    for word in word_list:
        if word == 'fking':
            newlist.append('fucking')
        elif word == 'fk':
            newlist.append('fucking')
        elif word == 'hizbollah':
            newlist.append('hezbollah')
        elif word == 'muticultural':
            newlist.append('multicultural')
        elif word == 'whatbdo':
            newlist.append('what do')
        elif word == 'mcinness':
            newlist.append('mcinnes')
        elif word == 'fundelmendalist':
            newlist.append('fundamentalist')
        elif word == 'cacusion':
            newlist.append('caucasian')
        else:
            newlist.append(word)
    return newlist


def misspelled3(word_list):
    """
    Changes some misspelled words.

    Parameters:
    word_list (list): list of words for datapoint

    Returns:
    new_list (list): updated list of words for datapoint.
    """

    newlist = []
    for word in word_list:
        if word == 'puertoricans':
            newlist.append('puerto ricans')
        elif word == 'withba':
            newlist.append('with a')
        elif word == 'foxnew':
            newlist.append('fox news')
        elif word == 'blelow':
            newlist.append('below')
        elif word == 'fanclub':
            newlist.append('fan club')
        elif word == 'aswer':
            newlist.append('answer')
        else:
            newlist.append(word)
    return newlist


def corrections(data):
    """
    Runs functions 'misspelled1', 'misspelled2', and 'misspelled3'.

    Parameters:
    data: feature variable database

    Returns:
    data: feature variable database with updated 'comment_text' feature.
    """

    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: replace_three(x)))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: misspelled(x)))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: misspelled2(x)))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: misspelled3(x)))
    data = data.loc[~(data.comment_text.isna())]
    data = data.reset_index(drop=True)
    return data


def load_cleaned():
    """
    Loads cleaned data.

    Loads data, removes comments with null values after loading,
    updates data.

    Parameters:
    data: feature variable database

    Returns:
    feature: feature variable database with rows containing null values
             removed.
    target: target variable database with rows containing null values
            removed.
    """

    features = pd.read_csv('features.csv')
    target = pd.read_csv('target.csv')
    data = pd.concat([target, features], axis=1)
    data = data.loc[~(data.comment_text.isna())]
    data = data.reset_index(drop=True)
    features = data[['comment_text']]
    features.to_csv('features.csv', index=False)
    target = data.drop(columns=['comment_text'])
    target = f.reduce_mem_usage(target)
    target.to_csv('target.csv', index=False)
    return features, target


def clean1():
    """
    First part of data cleaning functions.

    Reduces overall dataset to the data with group
    annotations, reduces memory usage, splits to target
    and features, and uploads 'target' dataset.

    Parameters:
    none

    Returns:
    features: feature variable database.
    """

    data = pd.read_csv('train_bias.csv')
    data = f.reduce_mem_usage(data)
    data = annotated(data)
    data = identities(data)
    data = target_cols(data)
    target = data[['offensive', 'offensive_and_identity']]
    target.to_csv('target.csv', index=False)
    features = data.drop(columns=['offensive', 'offensive_and_identity'])
    return features


def clean():
    """
    Second part of data cleaning functions.

    Cleans comments for tokenizing. Saves 'features' dataset.
    Reloads 'features' and 'target' datasets, and removes any
    datapoints with null data.

    Parameters:
    none

    Returns:
    features: feature variable database.
    target: target database.
    """

    features = clean1()
    features = column_change(features)
    features = comments_obscene(features)
    features = comments_obscene2(features)
    features = clean_punct(features)
    features = punct_space(features)
    features = whitespace(features)
    features = tokenize(features)
    features = rejoin(features)
    features = replace_halfwords(features)
    features = tokenize(features)
    features = corrections(features)
    features = rejoin(features)
    features = whitespace(features)
    features.to_csv('features.csv', index=False)
    features, target = load_cleaned()
    return features, target
