import os
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from retry import retry
import Levenshtein
from tqdm import tqdm

# API_SECRET_KEY = "sk-xxx"
# BASE_URL = "xxx"
# os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
# os.environ["OPENAI_API_BASE"] = BASE_URL
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain_openai import ChatOpenAI
from tools.poe_chat_client import PoeChatOpenAI as ChatOpenAI
import logging
from datetime import datetime

LOG_DIR = "/Users/ruyi/Desktop/NAIP/v1_resource/TKPD/log"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_PATH = os.path.join(LOG_DIR, f"prompt_keyword_async_search_{timestamp}.log")
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def normalized_edit_distance(str1, str2):
     
    str1 = str1.strip().lower()
    str2 = str2.strip().lower()
    if not str1 and not str2:
        return 0.0
    if not str1 or not str2:
        return 1.0

     
    edit_distance = Levenshtein.distance(str1, str2)

     
    max_length = max(len(str1), len(str2))

     
    normalized_distance = edit_distance / max_length if max_length != 0 else 0
    return normalized_distance
usr_prompts = [
    # "Given the following title and abstract of the research paper, identify the core task or problem being addressed in few words. You MUST respond with the keyphrase ONLY in this format: xxx",
            #    "Based on the given title and abstract, what is the main focus or task of the research? Summarize it in a few words. You MUST respond with the keyphrase ONLY in this format: xxx",
               "Based on the title and abstract, determine the main area of study for the paper, focusing on a keyword that accurately represents the field. You MUST respond with the keyword ONLY in this format: xxx.",
               "Analyze the title and abstract provided to identify the central task or topic of the paper, which will be used as a keyword for searching related academic papers on Google Scholar. Avoid terms that are either too broad (such as 'deep learning' or 'computer vision') or too specific (such as certain model names, unless widely recognized.). You MUST respond with the keyword ONLY in this format: xxx",
               "Given the title and abstract below, determine the specific research field by focusing on the main application area and the key technology. You MUST respond with the keyword ONLY in this format: xxx.",
               "Identify the research field from the given title and abstract. You MUST respond with the keyword ONLY in this format: xxx",
               ]
@retry(tries=3,delay=2)
def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True, model="gpt-5.4-nano", temperature=0):
    if not sys_content:
        sys_content = (
            # "You are a profound researcher who is good at identifying the topic key phrase from paper's title and "
            # "abstract. Ensure that the topic key phrase precisely defines the research area within the article. For effective academic searching, such as on Google Scholar, the field should be specifically targeted rather than broadly categorized. For instance, use 'image classification' instead of the general 'computer vision' to enhance relevance and searchability of related literature."
            "You are an expert metadata annotator. "
            "You must return exactly one research keyword that matches the paper's main field. "
            "The keyword must be a concise, canonical term (e.g., 'question answering', "
            "'federated learning'). Do not add adjectives, methods, datasets, or explanations. "
            "Return the keyword in lowercase and nothing else."
            )
    if not usr_prompt:
        usr_prompt = ("Analyze the title and abstract provided to identify the central topic of the paper, which will be used as a keyword for searching related academic papers on Google Scholar. Avoid terms that are either too broad (such as 'deep learning' or 'computer vision') or too specific (such as obscure model names, unless widely recognized). Focus on a keyword that reflects the innovative aspect or core methodology of the study. You MUST respond with the keyword ONLY in this format: xxx")

    messages = [SystemMessage(content=sys_content)]

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
    robustness.''' if abstract else ''
    if extra_prompt:
        messages += [HumanMessage(content=f'''{usr_prompt}\n\n{extra_abs_content}'''), AIMessage(content='test-time augmentation')]

    content = f'''{usr_prompt}\n
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'
    messages.append(HumanMessage(content=content))
    # print(messages)

    chat = ChatOpenAI(model=model, temperature=temperature)

    return chat.batch([messages])[0].content
import csv
from multiprocessing import Pool
prompt = "Given the title and abstract below, determine the specific research field by focusing on the main application area and the key technology. You MUST respond with the keyword ONLY in this format: xxx."
def process_row(row, prompt=prompt):
    title, abs, GT_kwd = row[0], row[1], row[2]
    pred_kwd = get_chatgpt_field(title, abs, usr_prompt=prompt) # This should be replaced with the actual prediction logic
    # Assuming normalized_edit_distance is defined elsewhere

    ned = normalized_edit_distance(GT_kwd, pred_kwd)
    print(f'GT:{GT_kwd} \t Pred:{pred_kwd} \t Ned:{ned}')
    logger.info(f'GT:{GT_kwd} \t Pred:{pred_kwd} \t Ned:{ned}')
    return ned

def main():
    with open(r'/Users/ruyi/Desktop/NAIP/v1_resource/TKPD/TKPD.csv','r', newline='', encoding='gbk') as input_csvfile:
        reader = csv.reader(input_csvfile)
        rows = [row for row in reader if row and row[0].lower() != "title"]
    print(len(rows))
    logger.info("Total rows: %d", len(rows))
    # with Pool(12) as p:
    #     results = p.map(process_row, rows)
    prompt_variants = usr_prompts
    for current_prompt in prompt_variants:
        results = []
        # sanitized_prompt = current_prompt.replace("\n", " ")
        # desc = f"Processing rows ({sanitized_prompt[:40]}...)"
        for row in tqdm(rows, desc="Processing rows", unit="row"):
            results.append(process_row(row, prompt=current_prompt))

        average_distance = sum(results) / len(results) if results else 0
        # print(f"{prompt}: {average_distance}")
        print(f"{current_prompt}: {average_distance}")
        print("-" * 40)
        logger.info("%s: %f", current_prompt, average_distance)
        logger.info("-" * 40)

if __name__ == '__main__':
    main()
