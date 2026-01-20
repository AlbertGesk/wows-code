#!/usr/bin/env python3
import click
import pyterrier as pt
from pathlib import Path
from tirex_tracker import tracking, ExportFormat
from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
from tqdm import tqdm
import string


def get_retriever(index, model, numr_results):
    model = model.upper()

    if model == "BM25":
        return index.bm25(k1=1.2, b=0.5, num_results=numr_results)
    elif model == "DPH":
        return index.dph()
    elif model == "PL2":
        return index.pl2()
    elif model == "TF_IDF":
        return index.tf_idf()
    else:
        raise ValueError(f"Unsupported retrieval model: {model}")

def run_query_expansion(index, dataset, query_expansion, first_model, last_model, reformulation, num_results):
    topics = pt.datasets.get_dataset(f"irds:ir-lab-wise-2025/{dataset}").get_topics("title")

    if query_expansion == "Bo1":
        if reformulation == "reformulation":
            pipe = get_retriever(index, first_model, num_results) >> pt.rewrite.tokenise() >> pt.rewrite.Bo1QueryExpansion(index.index_ref()) >> get_retriever(index, last_model, num_results) >> pt.rewrite.reset()
        else:
            pipe = get_retriever(index, first_model, num_results) >> pt.rewrite.tokenise() >> pt.rewrite.Bo1QueryExpansion(index.index_ref()) >> get_retriever(index, last_model, num_results)
        return pipe(topics)
    elif query_expansion == "RM3":
        if reformulation == "reformulation":
            pipe = get_retriever(index, first_model, num_results) >> pt.rewrite.tokenise() >> pt.rewrite.RM3(index.index_ref()) >> get_retriever(index, last_model, num_results) >> pt.rewrite.reset()
        else:
            pipe = get_retriever(index, first_model, num_results) >> pt.rewrite.tokenise() >> pt.rewrite.RM3(index.index_ref()) >> get_retriever(index, last_model, num_results)
        return pipe(topics)
    else :
        pipe = pt.rewrite.tokenise() >> get_retriever(index, first_model)
        return pipe(topics)
        
def extract_text_of_document(doc, field):
    # ToDo: here one can make modifications to the document representations
    if field == "default_text":
        return doc.default_text()
    elif field == "title":
        return doc.title
    elif field == "description":
        return doc.description

def get_index(dataset, field, output_path):
    index_dir = output_path / "indexes" / f"{dataset}-on-{field}"
    if not index_dir.is_dir():
        print("Build new index")
        docs = []
        dataset = ir_datasets.load(f"ir-lab-wise-2025/{dataset}")

        for doc in tqdm(dataset.docs_iter(), "Pre-Process Documents"):
            docs.append({"docno": doc.doc_id, "text": extract_text_of_document(doc, field)})

        with tracking(export_file_path=index_dir / "index-metadata.yml", export_format=ExportFormat.IR_METADATA):
            pt.IterDictIndexer(str(index_dir.absolute()), meta={'docno' : 100}, verbose=True).index(docs)
    return pt.terrier.TerrierIndex(str(index_dir))


def run_retrieval(output, index, dataset, text_field_to_retrieve, query_expansion, first_model, last_model, reformulation, num_results):
    if query_expansion == "no-qe":
        tag = f"pyterrier-on-{text_field_to_retrieve}-with-{first_model}-num_results-{num_results}"
        description = f"This is a PyTerrier retriever with {first_model} retrieving on the {text_field_to_retrieve} text representation of the documents. {query_expansion} is used to add additional terms."
    elif query_expansion in {"Bo1", "RM3"}:
        tag = f"pyterrier-on-{text_field_to_retrieve}-with-{first_model}-{query_expansion}-{last_model}-with-{reformulation}-num_results-{num_results}"
        description = f"This is a PyTerrier retriever pipeline retrieving on the {text_field_to_retrieve} text representation of the documents. Pipeline: {first_model}-{query_expansion}-{last_model}-{reformulation} is used to add additional terms."
    else:
        tag = f"pyterrier-on-{text_field_to_retrieve}-with-{first_model}-num_results-{num_results}"
        description = f"This is a PyTerrier retriever with {first_model} retrieving on the {text_field_to_retrieve} text representation of the documents. {query_expansion} is used to add additional terms."
    
    target_dir = output / "runs" / dataset / tag
    target_file = target_dir / "run.txt.gz"

    if target_file.exists():
        return

    with tracking(export_file_path=target_dir / "ir-metadata.yml", export_format=ExportFormat.IR_METADATA, system_description=description, system_name=tag): 
        run = run_query_expansion(index, dataset, query_expansion, first_model, last_model, reformulation, num_results)

    pt.io.write_results(run, target_file)

@click.command()
@click.option("--dataset", type=click.Choice(["radboud-validation-20251114-training", "spot-check-20251122-training"]), required=True, help="The dataset.")
@click.option("--output", type=Path, required=False, default=Path("output"), help="The output directory.")
@click.option("--text-field-to-retrieve", type=click.Choice(["default_text", "title", "description"]), required=False, default="default_text", help="The text field of the documents on which to retrieve.")
@click.option("--query-expansion", type=click.Choice(["no-qe", "Bo1", "RM3"]), required=False, default="no-qe", help="The query expansion algorithm.")
@click.option("--first-model", type=click.Choice(["BM25", "DPH", "PL2", "TF_IDF"]), required=False, default="BM25", help="The first retrieval model in the pipeline.")
@click.option("--last-model", type=click.Choice(["BM25", "DPH", "PL2", "TF_IDF"]), required=False, default="BM25", help="The last retrieval model in the pipeline if query expansion is used.")
@click.option("--reformulation", type=click.Choice(["reformulation", "no-reformulation"]), required=False, default="no-reformulation", help="Decide if reformulation should be done after query expansion")
@click.option("--num_results", type=click.Choice(["10", "100", "1000"]), required=False, default="1000", help="The maximum number of results to return per query for the retrieval models.")
def main(dataset, text_field_to_retrieve, query_expansion, first_model, last_model, reformulation, output, num_results):
    ensure_pyterrier_is_loaded(is_offline=False)

    index = get_index(dataset, text_field_to_retrieve, output)
    run_retrieval(output, index, dataset, text_field_to_retrieve, query_expansion, first_model, last_model, reformulation, num_results)
    
if __name__ == '__main__':
    main()

