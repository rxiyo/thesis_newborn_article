import csv
import logging
import time
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools.poe_chat_client import PoeChatOpenAI as ChatOpenAI
from tqdm import tqdm

LOG_PATH = Path("logs/gpt_keyword_labeler_all.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert metadata annotator. "
            "You must return exactly one research keyword that matches the paper's main field. "
            "The keyword must be a concise, canonical term (e.g., 'question answering', "
            "'federated learning'). Do not add adjectives, methods, datasets, or explanations. "
            "Return the keyword in lowercase and nothing else."
)

USER_PROMPT = (
    "Analyze the title and abstract provided to identify the central topic of the paper, which will be used as a keyword for searching related academic papers on Google Scholar. Avoid terms that are either too broad (such as 'deep learning' or 'computer vision') or too specific (such as obscure model names, unless widely recognized). Focus on a keyword that reflects the innovative aspect or core methodology of the study. You MUST respond with the keyword ONLY in this format: xxx"
)
extra_abs_content = '''
    Given Title: Dynamic Test-Time Augmentation via Differentiable Functions
    Given Abstract: Distribution shifts, which often occur in the real world, degrade the
    accuracy of deep learning systems, and thus improving robustness is essential
    for practical applications. To improve robustness, we study an image
    enhancement method that generates recognition-friendly images without
    retraining the recognition model. We propose a novel image enhancement method,
    DynTTA, which is based on differentiable data augmentation techniques and
    generates a blended image from many augmented images to improve the recognition
    accuracy under distribution shifts. In addition to standard data augmentations,
    DynTTA also incorporates deep neural network-based image transformation, which
    further improves the robustness. Because DynTTA is composed of differentiable
    functions, it is directly trained with the classification loss of the
    recognition model. We experiment with widely used image recognition datasets
    using various classification models, including Vision Transformer and
    MLP-Mixer. DynTTA improves the robustness with almost no reduction in
    classification accuracy for clean images, which is a better result than the
    existing methods. Furthermore, we show that estimating the training time
    augmentation for distribution-shifted datasets using DynTTA and retraining the
    recognition model with the estimated augmentation significantly improves
    robustness.'''

def predict_keyword(
    title: str,
    abstract: str,
    model: str = "gpt-5.4-nano",
    temperature: float = 0.0,
    max_retries: int = 5,
    retry_delay: float = 5.0,
) -> str:
    chat = ChatOpenAI(model=model, temperature=temperature)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f'''{USER_PROMPT}\n\n{extra_abs_content}'''), 
        AIMessage(content='test-time augmentation'),
        HumanMessage(content=f'''{USER_PROMPT}\n
                Given Title: {title}\n
                Given Abstract: {abstract}
            '''),
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = chat.batch([messages])[0].content.strip()
            logger.info("Predicted keyword - title: %s | keyword: %s", title, response)
            return response
        except Exception as exc:  # pragma: no cover
            if attempt == max_retries:
                logger.exception("Exceeded retries for '%s'.", title)
                raise
            wait = retry_delay * attempt
            logger.warning(
                "Request failed (attempt %d/%d): %s. Sleeping %.1f s before retry.",
                attempt,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

def iter_rows(csv_path: Path) -> Iterable[dict]:
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row

def _load_processed_keys(
    output_csv: Path, title_key: str, abstract_key: str
) -> Set[Tuple[str, str]]:
    processed: Set[Tuple[str, str]] = set()
    if not output_csv.exists():
        return processed
    with output_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            processed.add(
                (row.get(title_key, "").strip(), row.get(abstract_key, "").strip())
            )
    return processed

def annotate_dataset(
    input_csv: Path,
    output_csv: Path,
    title_key: str = "title",
    abstract_key: str = "abstract",
    resume: bool = True,
) -> None:
    input_rows = list(iter_rows(input_csv))
    logger.info("Loaded %d rows from %s", len(input_rows), input_csv)

    processed_keys = (
        _load_processed_keys(output_csv, title_key, abstract_key) if resume else set()
    )
    remaining_rows = [
        row
        for row in input_rows
        if (row.get(title_key, "").strip(), row.get(abstract_key, "").strip())
        not in processed_keys
    ]
    logger.info("Resume mode: %s | remaining rows: %d", resume, len(remaining_rows))

    mode = "a" if resume and output_csv.exists() else "w"
    write_header = mode == "w"
    fieldnames = list(input_rows[0].keys()) + ["predicted_keyword"]

    with output_csv.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for row in tqdm(remaining_rows, desc="Annotating", unit="paper"):
            title = row.get(title_key, "").strip()
            abstract = row.get(abstract_key, "").strip()
            keyword = predict_keyword(title, abstract)
            row["predicted_keyword"] = keyword
            writer.writerow(row)
            fh.flush()

    logger.info("Finished annotation. Output saved to %s", output_csv)

if __name__ == "__main__":
    INPUT = Path("/Users/ruyi/Desktop/NAIP/dataset/cleaned_high_cited_papers.csv")  # 调整为你的实际输入路径
    OUTPUT = Path("/Users/ruyi/Desktop/NAIP/dataset/cleaned_high_cited_papers_keywords.csv")
    annotate_dataset(INPUT, OUTPUT, resume=True)
