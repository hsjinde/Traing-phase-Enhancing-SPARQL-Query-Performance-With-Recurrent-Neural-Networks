"""
Utilities for deriving training labels from SPARQL query triples.

Example:
    import pandas as pd
    from AutotagTarget import GetTarget

    train_data = pd.read_excel("Data/LCQUAD.xlsx")
    processed_data = GetTarget(train_data)
"""

import re
from collections import Counter


LABEL_TRANSITIONS = {
    "S P ?ans ": "A",
    "S P ?x ": "a",
    "?ans P O ": "B",
    "?x P O ": "b",
    "?ans P ?x ": "C",
    "?x P ?ans ": "c",
    "S P O ": "D",
}

# Backwards-compatible name used by the original script.
Labletrans = LABEL_TRANSITIONS

LABEL_SEQUENCE = {
    "A": ["A", "E", "H"],
    "a": ["a", "e", "h"],
    "B": ["B", "F", "I"],
    "b": ["b", "f", "i"],
    "C": ["C", "G", "J"],
    "c": ["c", "g", "j"],
    "D": ["D"],
}


def GetTarget(train_data):
    if "query" not in train_data.columns:
        raise KeyError("GetTarget expects a dataframe with a 'query' column.")

    train_data = train_data.copy()
    if "lable" not in train_data.columns:
        insert_at = min(2, len(train_data.columns))
        train_data.insert(insert_at, "lable", None)

    for index, sparql in train_data["query"].items():
        query_body = _extract_query_body(str(sparql))
        spo_pattern = GetSPO(query_body)
        normalized_pattern = Convert_question_mark(spo_pattern, train_data, index)
        raw_labels = get_target(normalized_pattern)
        train_data.at[index, "lable"] = get_lable(raw_labels)

    return train_data


def _extract_query_body(query):
    match = re.search(r"\{(.*)\}", query, re.S)
    return match.group(1) if match else query


def GetSPO(subquery):
    pattern = []
    variable_count = 0

    for token in subquery.split():
        if token in {".", ";"}:
            continue

        token = token.rstrip(".")
        if "?" in token:
            pattern.append(token)
        else:
            pattern.append(("S", "P", "O")[variable_count % 3])
        variable_count += 1

    return pattern


def Delrdftype(query):
    query = list(query)
    while "rdf:type" in query:
        type_index = query.index("rdf:type")
        del query[max(type_index - 1, 0) : type_index + 2]
    return query


def Getans(query):
    tokens = query.split()
    if len(tokens) >= 3 and tokens[0].upper() == "SELECT" and tokens[1].upper() == "DISTINCT" and "?" in tokens[2]:
        return tokens[2].strip("()")

    select_match = re.search(r"\bSELECT\b\s+(?:DISTINCT\s+)?(?:\([^)]*\)\s+)?(\?\w+)", query, re.I)
    if select_match:
        return select_match.group(1)

    variables = re.findall(r"\?\w+", query)
    return variables[0] if variables else None


def Convert_question_mark(query, train_data, index):
    variables = []
    for token in query:
        if "?" in token and token not in variables:
            variables.append(token)

    if not variables:
        return list(query)

    ans = variables[0]
    if len(variables) > 1:
        detected_ans = Getans(str(train_data.at[index, "query"]))
        if detected_ans in variables:
            ans = detected_ans

    normalized = []
    for token in query:
        if "?" in token:
            normalized.append("?ans" if token == ans else "?x")
        else:
            normalized.append(token)
    return normalized


def get_target(*args):
    if len(args) == 3:
        train_data, query, index = args
    elif len(args) == 1:
        train_data, query, index = None, args[0], None
    else:
        raise TypeError("get_target expects query or train_data, query, index.")

    labels = []
    for i in range(0, len(query) - 2, 3):
        pattern = f"{query[i]} {query[i + 1]} {query[i + 2]} "
        label = LABEL_TRANSITIONS.get(pattern)
        if label is not None:
            labels.append(label)

    if train_data is not None:
        train_data.at[index, "lable"] = labels

    return labels


def get_lable(lable):
    if not lable:
        return ["d"]

    counts = Counter(lable)
    new_lable = []

    for key, sequence in LABEL_SEQUENCE.items():
        for label in sequence[: counts.get(key, 0)]:
            new_lable.append(label)

    return new_lable or ["d"]
