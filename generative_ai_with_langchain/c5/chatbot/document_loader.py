from typing import Any, Union, List
from langchain_community.document_loaders import (PyPDFLoader, TextLoader,
                                                  UnstructuredEPubLoader, UnstructuredWordDocumentLoader)
from loguru import logger
import pathlib
from langchain.schema import Document
from langchain_community.document_loaders.base import BaseLoader


class EPubLoader(UnstructuredEPubLoader):
    def __init__(self, file_path: Union[str, List[str]], **kwargs):
        super().__init__(file_path=file_path, **kwargs, mode='elements', strategy='fast')


class DocumentLoaderException(Exception):
    pass


class DocumentLoader(object):
    """Loads in a document with a supported extension."""
    supported_extensions = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.epub': EPubLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader,
    }


def load_document(temp_filepath: str) -> List[Document]:
    ext = pathlib.Path(temp_filepath).suffix
    loader: BaseLoader = DocumentLoader.supported_extensions.get(ext)
    if not loader:
        raise DocumentLoaderException(f"Invalid extension type {ext}, cannot load this type of file")
    loader = loader(temp_filepath)
    docs = loader.load()
    logger.info(docs)
    return docs


