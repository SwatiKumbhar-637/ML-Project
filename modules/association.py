# association.py

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import gradio as gr

def generate_rules(df):
    basket = df.groupby(['Customer ID', 'Order ID', 'Sub-Category'])['Sub-Category'] \
               .count().unstack(fill_value=0)

    basket = basket.applymap(lambda x: 1 if x > 0 else 0)
    basket = basket[basket.sum(axis=1) > 0]

    frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
    rules = rules.sort_values('lift', ascending=False)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    return rules

def ui(df):
    rules = generate_rules(df)

    def format_rules(df_rules):
        df_display = df_rules.copy()
        df_display['antecedents'] = df_display['antecedents'].apply(lambda x: ', '.join(list(x)))
        df_display['consequents'] = df_display['consequents'].apply(lambda x: ', '.join(list(x)))
        return df_display

    def show_rules():
        formatted_rules = format_rules(rules)
        return formatted_rules.to_html(index=False)

    gr.Interface(
        fn=show_rules,
        inputs=[],
        outputs=gr.HTML(label="Association Rules Table"),
        title="Association Rules Viewer",
        description="Displays association rules with support, confidence, and lift values."
    ).launch()
