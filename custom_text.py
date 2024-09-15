import logging
from pathlib import Path
from typing import Iterator, Optional, Union

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

logger = logging.getLogger(__name__)


import logging
from pathlib import Path
from typing import Iterator, Optional, Union

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings

logger = logging.getLogger(__name__)

class TextLoader(BaseLoader):
    """Load text file."""

    def __init__(
        self,
        file_path: Union[str, Path],
        encoding: Optional[str] = None,
        autodetect_encoding: bool = True,
    ):
        """Initialize with file path."""
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def lazy_load(self) -> Iterator[Document]:
        """Load from file path."""
        text = ""
        encodings_to_try = [self.encoding] if self.encoding else []

        if self.autodetect_encoding:
            detected_encodings = detect_file_encodings(self.file_path)
            encodings_to_try.extend([enc.encoding for enc in detected_encodings])

        if not encodings_to_try:
            encodings_to_try = ['utf-8', 'latin-1', 'ascii']

        for encoding in encodings_to_try:
            try:
                with self.file_path.open(encoding=encoding) as f:
                    text = f.read()
                logger.debug(f"Successfully loaded file with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                logger.debug(f"Failed to load with encoding: {encoding}")
                continue
            except Exception as e:
                logger.error(f"Error loading {self.file_path}: {str(e)}")
                raise RuntimeError(f"Error loading {self.file_path}") from e

        if not text:
            raise RuntimeError(f"Unable to load {self.file_path} with any encoding")

        metadata = {"source": str(self.file_path)}
        yield Document(page_content=text, metadata=metadata)