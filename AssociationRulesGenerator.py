# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:59:05 2024

@author: ivirc
"""

import numpy as np
import pandas as pd

import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules 


df = pd.read_csv('Phishing_Legitimate_Training.csv')

binary_columns = ['AtSymbol','TildeSymbol','NumHash','NoHttps','RandomString','IpAddress',
                  'DomainInSubdomains','DomainInPaths','DoubleSlashInPath','EmbeddedBrandName','ExtFavicon','InsecureForms',
                  'RelativeFormAction','ExtFormAction','AbnormalFormAction','FrequentDomainNameMismatch','FakeLinkInStatusBar'
                  ,'RightClickDisabled','PopUpWindow','SubmitInfoToEmail','IframeOrFrame','MissingTitle','ImagesOnlyInForm',]
print(binary_columns)

# filter out any column that isn't binary
df_binary = df[binary_columns + ['CLASS_LABEL']]

# convert to boolean values T/F
df_binary = df_binary.astype(bool)

#Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df_binary, min_support=0.1, use_colnames=True)

#generate association rules
rules = association_rules(frequent_itemsets, num_itemsets=len(frequent_itemsets), metric="lift", min_threshold=1.0)

#Filter rules where the consequent is "CLASS_LABEL = 1
class_label_1_rules = rules[rules['consequents']== {'CLASS_LABEL'}]

#Print the association rules for CLASS_LABEL = 1 
with pd.option_context('display.max_rows',None, 'display.max_columns', None):
    print(class_label_1_rules)
