import argparse
import sys

import pandas as pd
from sklearn import metrics


"""
print("Classification Report:")
    print(metrics.classification_report(labels, predictions, target_names=classes, zero_division=0))
    # TODO currently get a broadcast error, fix ValueError: shape mismatch: objects cannot be broadcast to a single shape
    # print("Confusion Matrix:")
    # print(metrics.confusion_matrix(test_labels_, predictions_, labels=[label_encoder.classes_]))"
"""

def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('-r1', help='results 1')
    p.add_argument('--r2', help='results 2')
    p.add_argument('--comparison', choices=["between_models", "between_pairs"], default="between_pairs")
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    print(args)
    ## Confusion Matrix section
    df1 = pd.read_csv(args.r1)
    if args.r2:
        df2 = pd.read_csv(args.r2)
    classes = list(map(str, range(1, 6)))

    # Assumes that same bias cat in all of them or doesn't work -- also assumes already filtered to convergence only
    bias_types = set(df1["bias_type"].values)
    print(bias_types)
    for bt in bias_types:
        print(bt)
        if bt == "rank":
            continue
        subselect1 = df1[df1["bias_type"] == bt]
        cat1, cat2 = subselect1["bias_cat_1"].sample().values[0], subselect1["bias_cat_2"].sample().values[0]

        if args.comparison == "between_models":
            assert args.r2, "need second set of results to compare between models"
            subselect2 = df2[df2["bias_type"] == bt]
            #cat 1
            print(cat1)
            pred1, pred2 = subselect1["label_1"].values, subselect2["label_1"].values
            print(metrics.classification_report(pred1, pred2, target_names=classes, zero_division=0))
            print(metrics.confusion_matrix(pred1, pred2))

            # cat 2
            print(cat2)
            pred1, pred2 = subselect1["label_2"].values, subselect2["label_2"].values
            print(metrics.classification_report(pred1, pred2, target_names=classes, zero_division=0))
            print(metrics.confusion_matrix(pred1, pred2))

        elif args.comparison == "between_pairs":
            print(f"between {cat1} and {cat2}")
            pred1, pred2 = subselect1["label_1"].values, subselect1["label_2"].values
            print(
                metrics.classification_report(pred1, pred2, target_names=classes, zero_division=0))
            print(metrics.confusion_matrix(pred1, pred2))





    """
    Types of qualitative analysis
    # By Seed
    Find examples that different seeds disagree on. 
     - Do they have anything in common? Are they all one type of gold label?
       * Parse logs for the indices that are disagreed on, and look up their gold label. Also see if
       they are variations on the same sentence template.
     - Are there examples where seeds always disagree at all points in the training?
        * parse logs to see if any indices appear in ~10 checkpoints
     - What about where they disagree early in the training but not late, or vice versa?
        * parse first half/last half of log and see if any indices appear in >=5 checkpoints of one 
        but not of the other.
        
    - Are these patterns different for boundary crossing disagreements?
    * categorise the disagreements as boundary crossing or not and filter by that 
     
     # Across Models
     - Is the difference between models driven by a few examples, or by a small difference in everything?
        * Find what percentage of examples two models agree on, for a given language -- TODO do this with a confusion matrix
        # TODO Start by generating a confusion matrix of all labels. 
        Then for both type separately for a type of bias 
        
        * use some kind of distance metric for their predictions (wasserstein)
        *** What does this mean in either case? Are there differences by bias type?
        
        ***Pseudo
        For each bias type:
            For each model (at convergence):
                for bias subtypes:
                    get_ordered_list_of_predictions()
                    concat subtypes
                    store
            compare all prediction lists across models (% in agreement, other distance metric?)
            print out indices and strings of disagreements > threshold
            
     - Are there examples that are treated differently by all models?
      * get list of paired predictions for each counterfactual, by model, and highlight either a) 
      boundary crossing changes across models or b) changes above some threshold
      
      PICK UP HERE
      
      - Are extremes of bias the same across models?
      * Find counterfactual pairs with big gaps, and see if it is the same across models. Get a
      list of high bias for each model, and then take intersection. 
     
     # Misc
     ** Generate everything both with intensity changes and also with boundary flipping
    
    
    Should be able to start from a results file and a corpus file and coindex them. 
    
    functions:
    get_ordered_list_of_predictions
    
    get_ordered_list_of_gaps
    
    
    
    
    NOTES:
    situation where maybe one model just mainly produces low numbers and one produces a big spread
    could be the reason for some differences in bias. Basically how conservative the model is in its choices/
    how big a range that is.
    
    Maybe need to use wasserstein distance etc between the distributions of the labels - can look at the labels in aggregate, see what the range 
    of what my model produces is
    
    
    ** Check if the English Amazon dataset is balanced or not? Or if the model can learn central tendency bias (or the reverse)
    ** Also don't just look at examples where the models disagee across many of them, but also where two models disagree by a lot
    
    *** Can a model learn central tendency bias even with balanced data? depends on the loss function -- and if its penalised for being wrong or being 
    just close to the answer.
    
    """