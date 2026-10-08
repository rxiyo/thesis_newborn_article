class Arxiv_paper:
    def __init__(self, result, ref_type="entity"):
        self.result = result
        self.id = get_arxiv_id_from_url(result.entry_id)  # 或者 result.entry_id.split('/')[-1]
        self.title = result.title
        self.pub_url = result.entry_id  # https://arxiv.org/abs/...
        self.ref_type = ref_type

    def _get_default_filename(self):
        return f"{self.id}.pdf"
    
def get_arxiv_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1]