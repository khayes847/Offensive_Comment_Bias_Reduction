#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Oct 8 10:50 2019

@author: khayes847

Collected functions for data cleaning.
"""
import re
# pylint: disable=unused-import
import swifter
# pylint: enable=unused-import
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import functions as f


# pylint: disable=unnecessary-lambda
def change(data):
    """Drops unnecessary columns, alters data labels"""
    data = data.loc[~(data.asian.isna())]
    data = data.loc[~(data.comment_text.isna())]
    data = data.drop(columns=['parent_id', 'article_id', 'created_date',
                              'publication_id'])
    data = data.rename(columns={'rating': 'cc_rejected', 'funny': 'cc_funny',
                                'wow': 'cc_wow', 'sad': 'cc_sad', 'likes':
                                'cc_likes', 'disagree': 'cc_disagree',
                                'identity_annotator_count':
                                'cc_identity_annotator_count',
                                'toxicity_annotator_count':
                                'cc_toxicity_annotator_count'})
    data['cc_rejected'] = (data['cc_rejected'].replace({'approved': 0,
                                                        'rejected': 1}))
    return data


def target_grouping(row):
    """Changes target based on grouping"""
    val = 0
    if row['no_group'] == 1:
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
    """Categorizes whether identity is mentioned in a comment, and
    whether it is offensive"""
    identity_list = list((data.iloc[:, 8:32]).columns)
    for iden in identity_list:
        data[f'{iden}'] = data[iden].swifter.apply(lambda x: x if x >= 0.15
                                                   else 0)
    data['groups'] = data[identity_list].sum(axis=1)
    data['no_group'] = data.groups.swifter.apply(lambda x: 1 if x == 0
                                                 else 0)
    data['max_group'] = data[identity_list].idxmax(axis=1)
    data_no = data.loc[data.no_group == 1]
    data_yes = data.loc[data.no_group == 0]
    data_no['max_group'] = 'none'
    data = pd.concat([data_no, data_yes], ignore_index=False)
    data['target'] = data.target.swifter.apply(lambda x: 1 if x >= .5 else 0)
    data['target_new'] = data.swifter.apply(lambda row:
                                            target_grouping(row), axis=1)
    return data


def reorder(data):
    """Reorders data columns"""
    col_list = ['id', 'target', 'target_new', 'cc_rejected',
                'cc_toxicity_annotator_count', 'cc_identity_annotator_count',
                'cc_likes', 'cc_disagree', 'cc_funny', 'cc_sad', 'cc_wow',
                'comment_text', 'severe_toxicity', 'obscene',
                'identity_attack', 'insult', 'threat', 'sexual_explicit',
                'max_group']
    data = data[col_list]
    return data


def comments_obscene(data):
    """Cleans words with punctuation obscuring obscenity"""
    data['comment_text'] = data['comment_text'].swifter.apply(lambda x:
                                                              x.lower())
    data['comment_text'] = data['comment_text'].str.replace(r'a\*\*\*\*\*e',
                                                            'asshole')
    data['comment_text'] = data['comment_text'].str.replace(r'a\*\*', 'ass')
    data['comment_text'] = data['comment_text'].str.replace(r'A\*\*', 'ASS')
    data['comment_text'] = data['comment_text'].str.replace(r'sh\*t', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'Sh\*t', 'Shit')
    data['comment_text'] = data['comment_text'].str.replace(r'SH\*T', 'SHIT')
    data['comment_text'] = data['comment_text'].str.replace(r'b\*tch', 'bitch')
    data['comment_text'] = data['comment_text'].str.replace(r'B\*tch', 'Bitch')
    data['comment_text'] = data['comment_text'].str.replace(r'B\*TCH', 'BITCH')
    data['comment_text'] = data['comment_text'].str.replace(r'f\*ck', 'fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'F\*ck', 'Fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'F\*CK', 'FUCK')
    data['comment_text'] = data['comment_text'].str.replace(r'f\*ggot',
                                                            'faggot')
    data['comment_text'] = data['comment_text'].str.replace(r'F\*ggot',
                                                            'Faggot')
    data['comment_text'] = data['comment_text'].str.replace(r'sl\*t', 'slut')
    data['comment_text'] = data['comment_text'].str.replace(r'Sl\*t', 'Slut')
    data['comment_text'] = data['comment_text'].str.replace(r'pr\*ck', 'prick')
    data['comment_text'] = data['comment_text'].str.replace(r'Pr\*ck', 'Prick')
    data['comment_text'] = data['comment_text'].str.replace(r'p\*ssy', 'pussy')
    data['comment_text'] = data['comment_text'].str.replace(r'P\*ssy', 'Pussy')
    data['comment_text'] = data['comment_text'].str.replace(r'P\*SSY', 'PUSSY')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'pu\$\$y', 'pussy'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'Pu\$\$y', 'Pussy'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'PU\$\$Y', 'PUSSY'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'a\$\$hole', 'asshole'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'A\$\$hole', 'Asshole'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'A\$\$HOLE', 'ASSHOLE'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'a\$\$', 'ass'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'A\$\$', 'ASS'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'du\*b', 'dumb'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\*\*\*\*\*g', 'fucking'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'm\*\*\*\*m', 'muslim'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\*\*\*\*d', 'fucked'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\*\*\*\*\*\*', 'fucking'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'F\*\*\*\*\*\*', 'Fucking'))
    data['comment_text'] = data['comment_text'].str.replace(r'f\*\*\*', 'fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'F\*\*\*', 'Fuck')
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.lower()))
    return data


def comments_obscene2(data):
    """Cleans words with punctuation obscuring obscenity"""
    data['comment_text'] = data['comment_text'].swifter.apply(lambda x:
                                                              x.lower())
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'b\*\*\*\*\*\*\*', 'bullshit'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'b\*\*\*\*\*\*t', 'bullshit'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'B\*\*\*\*\*\*T', 'BULLSHIT'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'n\*gger', 'nigger'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r' n\*\*\*\*\*', ' nigger'))
    data['comment_text'] = (data['comment_text'].str.replace
                            (r' N\*\*\*\*\*', ' Nigger'))
    data['comment_text'] = data['comment_text'].str.replace(r'g\*y', 'gay')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'f\@gg0t', 'faggot'))
    data['comment_text'] = data['comment_text'].str.replace(r'f\@g', 'fag')
    data['comment_text'] = (data['comment_text'].str.replace
                            (' ahole ', ' asshole '))
    data['comment_text'] = (data['comment_text'].str.replace
                            (' Ahole ', ' Asshole '))
    data['comment_text'] = (data['comment_text'].str.replace
                            (' AHOLE ', ' ASSHOLE '))
    data['comment_text'] = data['comment_text'].str.replace('http', '')
    data['comment_text'] = data['comment_text'].str.replace('HTTP', '')
    data['comment_text'] = data['comment_text'].str.replace('Http', '')
    data['comment_text'] = data['comment_text'].str.replace(r'f\*\*k', 'fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'F\*\*k', 'Fuck')
    data['comment_text'] = data['comment_text'].str.replace(r'F\*\*K', 'FUCK')
    data['comment_text'] = data['comment_text'].str.replace(r'h\*\*l', 'hell')
    data['comment_text'] = data['comment_text'].str.replace(r'H\*\*L', 'HELL')
    data['comment_text'] = data['comment_text'].str.replace(r'h\*ll', 'hell')
    data['comment_text'] = data['comment_text'].str.replace(r'H\*LL', 'HELL')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'He\*l H\*tler', 'Heil Hitler'))
    data['comment_text'] = data['comment_text'].str.replace(r's\*\*\*', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'S\*\*\*', 'Shit')
    data['comment_text'] = data['comment_text'].str.replace(r's\*\*t', 'shit')
    data['comment_text'] = data['comment_text'].str.replace(r'S\*\*t', 'Shit')
    data['comment_text'] = data['comment_text'].str.replace(r'S\*\*T', 'SHIT')
    data['comment_text'] = data['comment_text'].str.replace(r'an\*l', 'anal')
    data['comment_text'] = data['comment_text'].str.replace(r's\*x', 'sex')
    data['comment_text'] = data['comment_text'].str.replace(r'S\*x', 'Sex')
    data['comment_text'] = data['comment_text'].str.replace(r'c\*nt', 'cunt')
    data['comment_text'] = data['comment_text'].str.replace(r'C\*nt', 'cunt')
    data['comment_text'] = data['comment_text'].str.replace(r'tw\*t', 'twat')
    data['comment_text'] = data['comment_text'].str.replace(r'Tw\*t', 'Twat')
    data['comment_text'] = data['comment_text'].str.replace(r'wh\*re', 'whore')
    data['comment_text'] = data['comment_text'].str.replace(r'Wh\*re', 'Whore')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'nigg\*r', 'nigger'))
    data['comment_text'] = data['comment_text'].str.replace(r'c\*unt', 'cunt')
    data['comment_text'] = (data['comment_text'].str.replace
                            (r'pu\#\#y', 'pussy'))
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: x.lower()))
    return data


def clean_punct(data):
    """Removes non-stopping punctuation"""
    punct_list = [r'\-', r'\_', r"\'", r'\\', r'\/', r'\,', r'\*', r"\:",
                  r"\;", r"\(", r"\)", r"\{", r"\}", r"\[", r"\]", r"\|",
                  r"\<", r"\>", r"\#", r"\@", r'\%', r'\^', r'\+', r'\=',
                  r'\"', r'\&']
    for punct in punct_list:
        data['comment_text'] = data.comment_text.str.replace(f'{punct}', " ")
    return data


def punct_space(data):
    """Puts space between stopping punctuation and letters"""
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    punct_list = [r'\.', r'\?', r'\!']
    for punct in punct_list:
        for let in alphabet:
            data['comment_text'] = (data.comment_text.str.replace
                                    (f'{punct}{let}', f'{punct} {let}'))
    return data


def whitespace(data):
    """Reduces whitespace in comments"""
    i = 0
    while i <= 10:
        data['comment_text'] = data.comment_text.str.replace('  ', ' ')
        i += 1
    return data


def tokenize(data):
    """Tokenizes data"""
    word_tokenizer = RegexpTokenizer(r'\w+')
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: word_tokenizer.tokenize(x)))
    return data


def rejoin(data):
    """Rejoins tokenized data"""
    data['comment_text'] = (data['comment_text'].swifter.apply
                            (lambda x: ' '.join(x)))
    return data


def replace_halfwords(data):
    """Replaces part words from tokenized words"""
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
    """Replaces letters that appear more than three times in a row"""
    pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
    new_list = []
    for word in word_list:
        new_word = str(pattern.sub(r"\1\1", word))
        new_list.append(new_word)
    return new_list


def misspelled(x_list):
    """Changes some misspelled words"""
    newlist = []
    for x_val in x_list:
        if x_val == 'bicthes':
            newlist.append('bitches')
        elif x_val == 'fukin':
            newlist.append('fucking')
        elif x_val == 'fectless':
            newlist.append('feckless')
        elif x_val == 'trustworhiness':
            newlist.append('trustworthiness')
        elif x_val == 'awfucl':
            newlist.append('awful')
        elif x_val == 'sextiimg':
            newlist.append('sexting')
        elif x_val == 'licemse':
            newlist.append('license')
        else:
            newlist.append(x_val)
    return newlist


def misspelled2(x_list):
    """Changes some misspelled words"""
    newlist = []
    for x_val in x_list:
        if x_val == 'fking':
            newlist.append('fucking')
        elif x_val == 'fk':
            newlist.append('fucking')
        elif x_val == 'hizbollah':
            newlist.append('hezbollah')
        elif x_val == 'muticultural':
            newlist.append('multicultural')
        elif x_val == 'whatbdo':
            newlist.append('what do')
        elif x_val == 'mcinness':
            newlist.append('mcinnes')
        elif x_val == 'fundelmendalist':
            newlist.append('fundamentalist')
        elif x_val == 'cacusion':
            newlist.append('caucasian')
        else:
            newlist.append(x_val)
    return newlist


def misspelled3(x_list):
    """Changes some misspelled words"""
    newlist = []
    for x_val in x_list:
        if x_val == 'puertoricans':
            newlist.append('puerto ricans')
        elif x_val == 'withba':
            newlist.append('with a')
        elif x_val == 'foxnew':
            newlist.append('fox news')
        elif x_val == 'blelow':
            newlist.append('below')
        elif x_val == 'fanclub':
            newlist.append('fan club')
        elif x_val == 'aswer':
            newlist.append('answer')
        else:
            newlist.append(x_val)
    return newlist


def corrections(data):
    """Corrects some misspellings"""
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


def clean():
    """Performs data cleaning functions and splits data into
    smaller datasets"""
    data = f.open_s3('thisa', 'train_bias.csv')
    data = f.reduce_mem_usage(data)
    data = change(data)
    data = identities(data)
    data = reorder(data)
    data = comments_obscene(data)
    data = comments_obscene2(data)
    data = clean_punct(data)
    data = punct_space(data)
    data = whitespace(data)
    data = tokenize(data)
    data = rejoin(data)
    data = replace_halfwords(data)
    data = tokenize(data)
    data = corrections(data)
    data = rejoin(data)
    data = whitespace(data)
    data.to_csv('data_cleaned.csv', index=False)
    f.upload_s3('data_cleaned.csv', 'thisa', 'data_cleaned.csv')
    return data
