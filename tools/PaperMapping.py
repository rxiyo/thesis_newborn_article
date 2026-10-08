from sqlalchemy import create_engine, Column, Integer, String, Boolean
from tools.Arxiv_paper import Arxiv_paper

class PaperMapping(Base):
    __tablename__ = "paper_mapping"

    id = Column(String, primary_key=True)          # 直接用 arxiv ID
    title = Column(String, nullable=False)
    pub_url = Column(String, nullable=False)
    search_by_keywords = Column(String, nullable=True)
    has_been_downloaded = Column(Boolean, default=False)

    def __init__(self, arxiv_paper: Arxiv_paper, search_by_keywords: str):
        self.id = arxiv_paper.id
        self.title = arxiv_paper.title
        self.pub_url = arxiv_paper.pub_url
        self.search_by_keywords = search_by_keywords