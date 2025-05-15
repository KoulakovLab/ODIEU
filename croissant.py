import mlcroissant as mlc
import json


# FileObjects and FileSets define the resources of the dataset.
distribution = [
    # ODIEU is hosted on a GitHub repository:
    mlc.FileObject(
        id="github-repository",
        name="odieu-repository",
        description="ODIEU repository on github",
        content_url="https://github.com/KoulakovLab/ODIEU/",
        encoding_formats=["git+https"],
        sha256="main",
    ),
   # Within that repository, a FileSet lists all zip files:
   mlc.FileSet(
       id="model-archives",
       name="model-archives",
       description="Zipped archives of the models trained for the benchmark",
       contained_in=["github-repository"],
       encoding_formats=["application/zip"],
       includes="models/*.zip",
   ),
   # Within that repository, a FileSet lists all csv files:
   mlc.FileSet(
       id="data-files",
       name="data-files",
       description="100'000 olfactory descriptions dataset.",
       contained_in=["github-repository"],
       encoding_formats=["text/csv"],
       includes="datasets/*.csv",
   ),
]
#record_sets = [
#    # RecordSets contains records in the dataset.
#    mlc.RecordSet(
#        id="jsonl",
#        name="jsonl",
#        # Each record has one or many fields...
#        fields=[
#            # Fields can be extracted from the FileObjects/FileSets.
#            mlc.Field(
#                id="jsonl/context",
#                name="context",
#                description="",
#                data_types=mlc.DataType.TEXT,
#                source=mlc.Source(
#                    file_set="jsonl-files",
#                    # Extract the field from the column of a FileObject/FileSet:
#                    extract=mlc.Extract(column="context"),
#                ),
#            ),
#            mlc.Field(
#                id="jsonl/completion",
#                name="completion",
#                description="The expected completion of the promt.",
#                data_types=mlc.DataType.TEXT,
#                source=mlc.Source(
#                    file_set="jsonl-files",
#                    extract=mlc.Extract(column="completion"),
#                ),
#            ),
#            mlc.Field(
#                id="jsonl/task",
#                name="task",
#                description=(
#                    "The machine learning task appearing as the name of the"
#                    " file."
#                ),
#                data_types=mlc.DataType.TEXT,
#                source=mlc.Source(
#                    file_set="jsonl-files",
#                    extract=mlc.Extract(
#                        file_property=mlc._src.structure_graph.nodes.source.FileProperty.filename
#                    ),
#                    # Extract the field from a regex on the filename:
#                    transforms=[mlc.Transform(regex="^(.*)\\.jsonl$")],
#                ),
#            ),
#        ],
#    )
#]

# Metadata contains information about the dataset.
metadata = mlc.Metadata(
    name="gpt-3",
    # Descriptions can contain plain text or markdown.
    description=(
        "Recent advances in deep learning models for odorant perception prediction are opening new avenues for odor classification and generation. However, current classifiers are limited to predicting percepts from a fixed vocabulary and fail to capture the full richness of olfactory experience, including nuanced gradations between percepts. Progress in this area is hindered by the lack of large-scale olfactory datasets and the absence of standardized metrics for evaluating the quality of natural language smell descriptions—both of which are critical for training next-generation generative models. To address these limitations, we introduce Odor Description and Inference Evaluation Understudy (ODIEU), a novel benchmark comprising over 10,000 molecules and a model-based metric for evaluating generative models on both constrained and open-ended descriptions of molecular percepts. ODIEU fills a crucial gap in current evaluation methodologies for olfactory perception in the era of Large Language Models (LLMs) and provides a platform to accelerate the development of foundation models for olfaction. We demonstrate that general-purpose pretrained LLMs, across a range of sizes, lack the specialized domain knowledge required to accurately differentiate between olfactory percepts described using natural language. To overcome this limitation, we propose a model-based approach in which pretrained Sentence-BERT models are fine-tuned on olfactory descriptions using contrastive learning objectives. This significantly improves the separability of human-generated descriptions across different molecules. Overall, ODIEU offers a standardized framework for evaluating computational models of molecular perception, highlighting both the strengths and limitations of current approaches, and establishing a foundation for future progress in olfactory modeling, percept description, and evaluation."
    ),
    cite_as=(
        ""
    ),
    url="https://github.com/KoulakovLab/ODIEU/",
    distribution=distribution,
    # record_sets=record_sets,
)


with open("croissant.json", "w") as f:
  content = metadata.to_json()
  content = json.dumps(content, indent=2)
  print(content)
  f.write(content)
  f.write("\n")  # Terminate file with newline

dataset = mlc.Dataset(jsonld="croissant.json")
# records = dataset.records(record_set="jsonl")

for i, record in enumerate(records):
  print(record)
  if i > 10:
    break

