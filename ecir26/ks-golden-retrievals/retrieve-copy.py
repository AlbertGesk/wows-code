#!/usr/bin/env python3
import click
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import nltk

def run_query_expansion(index, dataset, retrieval_model, query_expansion):
    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset}").get_topics("title")
    
    if query_expansion == "no-qe":
        return topics
    elif query_expansion == "bo1":
        bo1_expansion = pt.BatchRetrieve(index, wmodel=retrieval_model) >> pt.rewrite.Bo1QueryExpansion(index)
        return bo1_expansion(topics)
    elif query_expansion == "RM3":
        rm3_expansion = pt.BatchRetrieve(index, wmodel=retrieval_model) >> pt.rewrite.RM3(index) >> pt.BatchRetrieve(index, wmodel=retrieval_model)
        return rm3_expansion(topics)

def extract_text_of_document(doc, field):
    # ToDo: here one can make modifications to the document representations
    if field == "default_text":
        return doc.default_text()
    elif field == "title":
        return doc.title
    elif field == "description":
        return doc.description


def get_index(dataset, field, output_path, query_expansion):
    index_dir = output_path / "indexes" / f"{dataset}-on-{field}-with-{query_expansion}"
    if not index_dir.is_dir():
        print("Build new index")
        docs = []
        dataset = ir_datasets.load(f"ir-lab-wise-2025/{dataset}")

        for doc in tqdm(dataset.docs_iter(), "Pre-Process Documents"):
            docs.append({"docno": doc.doc_id, "text": extract_text_of_document(doc, field)})

        with tracking(export_file_path=index_dir / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
            pt.IterDictIndexer(str(index_dir.absolute()), meta={'docno' : 100}, verbose=True).index(docs)

    return pt.IndexFactory.of(str(index_dir.absolute()))


def run_retrieval(output, index, dataset, retrieval_model, text_field_to_retrieve, query_expansion):
    tag = f"pyterrier-{retrieval_model}-on-{text_field_to_retrieve}-with-{query_expansion}"
    target_dir = output / "runs" / dataset / tag
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        return

    
    topics = run_query_expansion(index, dataset, retrieval_model, query_expansion)

    retriever = pt.terrier.Retriever(index, wmodel=retrieval_model)

    description = f"This is a PyTerrier retriever using the retrieval model {retriever} retrieving on the {text_field_to_retrieve} text representation of the documents. Everything is set to the defaults."

    with tracking(export_file_path=target_dir / "ir-metadata.yml", export_format=ExportFormat.IR_METADATA, system_description=description, system_name=tag):
        run = retriever(topics)

    pt.io.write_results(run, target_file)









@click.command()
@click.option("--dataset", type=click.Choice(["radboud-validation-20251114-training", "spot-check-20251122-training"]), required=True, help="The dataset.")
@click.option("--output", type=Path, required=False, default=Path("output"), help="The output directory.")
@click.option("--retrieval-model", type=str, default="BM25", required=False, help="The retrieval model (e.g., BM25, PL2, DirichletLM).")
@click.option("--text-field-to-retrieve", type=click.Choice(["default_text", "title", "description"]), required=False, default="default_text", help="The text field of the documents on which to retrieve.")
@click.option("--query-expansion", type=click.Choice(["no-qe", "bo1", "RM3"]), required=False, default="no-qe", help="The query expansion algorithm.")
def main(dataset, text_field_to_retrieve, retrieval_model, query_expansion, output):
    ensure_pyterrier_is_loaded(is_offline=False)

    index = get_index(dataset, text_field_to_retrieve, output, query_expansion)
    run_retrieval(output, index, dataset, retrieval_model, text_field_to_retrieve, query_expansion)
    

if __name__ == '__main__':
    main()

