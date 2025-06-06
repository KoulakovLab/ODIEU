# ODIEU
Odor Description and Inference Evaluation Understudy 

### Abstract
Recent advances in deep learning models for odorant perception prediction are opening new avenues for odor classification and generation. However, current classifiers are limited to predicting percepts from a fixed vocabulary and fail to capture the full richness of olfactory experience, including nuanced gradations between percepts. Progress in this area is hindered by the lack of large-scale olfactory datasets and the absence of standardized metrics for evaluating the quality of natural language smell descriptions—both of which are critical for training next-generation generative models. To address these limitations, we introduce Odor Description and Inference Evaluation Understudy (ODIEU), a novel benchmark comprising over 10,000 molecules and a model-based metric for evaluating generative models on both constrained and open-ended descriptions of molecular percepts. ODIEU fills a crucial gap in current evaluation methodologies for olfactory perception in the era of Large Language Models (LLMs) and provides a platform to accelerate the development of foundation models for olfaction. We demonstrate that general-purpose pretrained LLMs, across a range of sizes, lack the specialized domain knowledge required to accurately differentiate between olfactory percepts described using natural language. To overcome this limitation, we propose a model-based approach in which pretrained Sentence-BERT models are fine-tuned on olfactory descriptions using contrastive learning objectives. This significantly improves the separability of human-generated descriptions across different molecules. Overall, ODIEU offers a standardized framework for evaluating computational models of molecular perception, highlighting both the strengths and limitations of current approaches, and establishing a foundation for future progress in olfactory modeling, percept description, and evaluation.

### Content
This repository contains the Sentence BERT models finetuned on quality olfactory data, as well as the dataset of 10'124 human-authored olfactory descriptions.

###### Models
We finetuned two publicly available LLM for sentence embeddings:
1. A smaller with 33.4M parameters: [all-MiniLM-L12-v1](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v1)
2. A larger with 435M parameters: [stella_en_400M_v5](https://huggingface.co/NovaSearch/stella_en_400M_v5)

The trained models are available under [models]([https://github.com/KoulakovLab/ODIEU/models/) folder.

###### Human-authored olfactory percept descriptions
Our dataset of 10'124 publicly available olfactory percept descriptions has been aggregated from two different sources:
1. Leffingwell and Arctander from the [pyfurme](https://github.com/pyrfume/pyrfume-data) repository
2. Goodscents from our own aggregation of the data present in [The Goodscents Company](https://www.thegoodscentscompany.com/misc/about.html) information system.

We call the three datasets combined LAG, it is available in the [datasets](https://github.com/KoulakovLab/ODIEU/datasets/) folder.

### How to use
An example of how to load one of the models and compute the embedding of olfcatory descriptions can be found in the [example](https://github.com/KoulakovLab/ODIEU/example.py).
