# Methodological Position

## Purpose

This project is not a demonstration that K-means is an effective production model for consumer complaint narratives. It is a case study in the relationship between **corpus context**, **problem formulation**, **text representation**, and **model behaviour**.

A common modelling error is to select an algorithm first and only later ask whether the structure of the data supports the assumptions implied by that algorithm. In text clustering, documents can contain multiple themes, reuse legal templates, mention company names repeatedly, and share a large domain vocabulary. These properties can prevent sharp separation even when preprocessing is technically careful.

## Why weak separation matters

The best tested silhouette score was approximately 0.0245. Values near zero indicate that documents are often almost as close to neighbouring clusters as they are to their assigned cluster. The two-dimensional projection also showed extensive overlap, although it retained only a small part of the original variance.

These findings are not hidden as a modelling failure. They show that the corpus and the chosen hard-clustering formulation do not support a strong claim of distinct topic groups.

## What preprocessing can and cannot do

Preprocessing can remove obvious noise, standardize text, control rare and ubiquitous terms, and construct a consistent numerical representation. It cannot turn a multi-topic document into a single-topic document, remove substantive overlap, guarantee spherical clusters, prevent legal templates from dominating lexical similarity, or determine the correct analytical objective.

## Broader lesson

A text-modelling workflow should first ask:

1. What is the unit of meaning: document, paragraph, sentence, or event?
2. Can one observation legitimately belong to several themes?
3. Are repeated templates being mistaken for substantive topics?
4. Is the goal exploration, taxonomy construction, retrieval, classification, or prediction?
5. What evidence would count as a useful result?

The answers may lead away from K-means toward sentence segmentation, soft clustering, topic models, embeddings, supervised learning, or qualitative analysis. The value of this project lies in making the limitation visible and showing why context-aware problem design must precede model construction.
