from typing import Callable
import os
import pandas as pd

from sklearn.preprocessing import scale

def read(name):
    f = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', name))
    return pd.read_csv(f, encoding='latin-1', index_col=0)

def remove_extras(df: pd.DataFrame) -> pd.DataFrame:
    """Removes the filename, x, y, w and h cols"""
    return df.drop(['filename','x', 'y', 'w', 'h'], axis=1)

def general(df: pd.DataFrame) -> pd.DataFrame:
    """Drops `specific_class` and renames `general_class` to `class`"""
    general = df.drop('specific_class', axis=1)
    general.rename(columns={"general_class": "class"},  inplace=True)
    return general

def specific(df: pd.DataFrame) -> pd.DataFrame:
    """Drops `general_class` and renames `specific_class` to `class`"""
    specific = df.drop('general_class', axis=1)
    specific.rename(columns={"specific_class": "class"},  inplace=True)
    return specific

def remove_below(n: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Returns a function which takes a dataframe and removes all classes whose count is less than the minimum specified"""
    def rem(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby('class').filter(lambda x: len(x) >= n)
    return rem

#downsampling
def balance_by_min(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a downsampled dataset."""
    if 'class' not in df.columns:
        raise ValueError('No class column present in DataFrame. Please apply the `general` or `specific` functions before.')
    g = df.groupby('class')
    size = g.size().min()
    return g.apply(lambda x: x.sample(size, random_state=0)).reset_index(drop=True)

def split_by_min(df: pd.DataFrame) -> tuple :
    """Returns a downsampled dataset and the remainder of the dataset."""
    if 'class' not in df.columns:
        raise ValueError('No class column present in DataFrame. Please apply the `general` or `specific` functions before.')
    g = df.groupby('class')
    size = g.size().min()
    sampled = g.apply(lambda x: x.sample(size, random_state=0))
    return sampled.reset_index(drop=True), df.drop(sampled.index.droplevel()).reset_index(drop=True)

def pipe(functions):
    """Takes a list of functions, and returns a function which runs all of them in order, piping the results."""
    def comp(*args):
        results = functions[0](*args)
        for f in functions[1:]:
            results = f(results)
        return results
    return comp
