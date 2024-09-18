# Traing phase Enhancing SPARQL Query Performance With Recurrent Neural Networks

## Environment

- python version : `3.7.16`
- requirements : pip3 install -r requirements.txt

## Multi-label Tagging

Extracts and tags query statements from LCQUAD datasets. The code implements the conversion of query statements into a specific labeling format and returns the processed dataset.


```mermaid
flowchart LR
    A[Get data] --> B[trans Query to SPO]
    B --> C[Labeling]
    C --> D[Label Mapping]
    D --> E[Get label]
```


