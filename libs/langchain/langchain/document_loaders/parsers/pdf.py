"""Module contains common parsers for PDFs."""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
)
from urllib.parse import urlparse

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.schema import Document

if TYPE_CHECKING:
    import pdfplumber.page


class PyPDFParser(BaseBlobParser):
    """Load `PDF` using `pypdf` and chunk at character level."""

    def __init__(self, password: Optional[Union[str, bytes]] = None):
        self.password = password

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf

        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            yield from [
                Document(
                    page_content=page.extract_text(),
                    metadata={"source": blob.source, "page": page_number},
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]


class PDFMinerParser(BaseBlobParser):
    """Parse `PDF` using `PDFMiner`."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        from pdfminer.high_level import extract_text

        with blob.as_bytes_io() as pdf_file_obj:
            text = extract_text(pdf_file_obj)
            metadata = {"source": blob.source}
            yield Document(page_content=text, metadata=metadata)


class PyMuPDFParser(BaseBlobParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        self.text_kwargs = text_kwargs or {}

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz

        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            yield from [
                Document(
                    page_content=page.get_text(**self.text_kwargs),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.number,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc
            ]


class PyPDFium2Parser(BaseBlobParser):
    """Parse `PDF` with `PyPDFium2`."""

    def __init__(self) -> None:
        """Initialize the parser."""
        try:
            import pypdfium2  # noqa:F401
        except ImportError:
            raise ImportError(
                "pypdfium2 package not found, please install it with"
                " `pip install pypdfium2`"
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdfium2

        # pypdfium2 is really finicky with respect to closing things,
        # if done incorrectly creates seg faults.
        with blob.as_bytes_io() as file_path:
            pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
            try:
                for page_number, page in enumerate(pdf_reader):
                    text_page = page.get_textpage()
                    content = text_page.get_text_range()
                    text_page.close()
                    page.close()
                    metadata = {"source": blob.source, "page": page_number}
                    yield Document(page_content=content, metadata=metadata)
            finally:
                pdf_reader.close()


class PDFPlumberParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
        self, text_kwargs: Optional[Mapping[str, Any]] = None, dedupe: bool = False
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self._process_page_content(page),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.page_number,
                            "total_pages": len(doc.pages),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc.pages
            ]

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)


class AmazonTextractPDFParser(BaseBlobParser):
    """Send `PDF` files to `Amazon Textract` and parse them.

    For parsing multi-page PDFs, they have to reside on S3.
    """

    def __init__(
        self,
        textract_features: Optional[Sequence[int]] = None,
        client: Optional[Any] = None,
    ) -> None:
        """Initializes the parser.

        Args:
            textract_features: Features to be used for extraction, each feature
                               should be passed as an int that conforms to the enum
                               `Textract_Features`, see `amazon-textract-caller` pkg
            client: boto3 textract client
        """

        try:
            import textractcaller as tc

            self.tc = tc
            if textract_features is not None:
                self.textract_features = [
                    tc.Textract_Features(f) for f in textract_features
                ]
            else:
                self.textract_features = []
        except ImportError:
            raise ImportError(
                "Could not import amazon-textract-caller python package. "
                "Please install it with `pip install amazon-textract-caller`."
            )

        if not client:
            try:
                import boto3

                self.boto3_textract_client = boto3.client("textract")
            except ImportError:
                raise ImportError(
                    "Could not import boto3 python package. "
                    "Please install it with `pip install boto3`."
                )
        else:
            self.boto3_textract_client = client

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Iterates over the Blob pages and returns an Iterator with a Document
        for each page, like the other parsers If multi-page document, blob.path
        has to be set to the S3 URI and for single page docs the blob.data is taken
        """

        url_parse_result = urlparse(str(blob.path)) if blob.path else None
        # Either call with S3 path (multi-page) or with bytes (single-page)
        if (
            url_parse_result
            and url_parse_result.scheme == "s3"
            and url_parse_result.netloc
        ):
            textract_response_json = self.tc.call_textract(
                input_document=str(blob.path),
                features=self.textract_features,
                boto3_textract_client=self.boto3_textract_client,
            )
        else:
            textract_response_json = self.tc.call_textract(
                input_document=blob.as_bytes(),
                features=self.textract_features,
                call_mode=self.tc.Textract_Call_Mode.FORCE_SYNC,
                boto3_textract_client=self.boto3_textract_client,
            )

        current_text = ""
        current_page = 1
        for block in textract_response_json["Blocks"]:
            if "Page" in block and not (int(block["Page"]) == current_page):
                yield Document(
                    page_content=current_text,
                    metadata={"source": blob.source, "page": current_page},
                )
                current_text = ""
                current_page = int(block["Page"])
            if "Text" in block:
                current_text += block["Text"] + " "

        yield Document(
            page_content=current_text,
            metadata={"source": blob.source, "page": current_page},
        )


class DocumentIntelligencePageParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Forms Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    def _generate_docs(self, blob: Blob, result: Any) -> Iterator[Document]:
        for p in result.pages:
            content = " ".join([line.content for line in p.lines])

            d = Document(
                page_content=content,
                metadata={
                    "source": blob.source,
                    "page": p.page_number,
                },
            )
            yield d

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()

            docs = self._generate_docs(blob, result)

            yield from docs


class DocumentNode:
    """A helper class to aggregate parse results into documents"""

    def __init__(
        self, type: str, role: str, start: int, end: int, content: str, page: int
    ) -> None:
        self.type = type
        self.role = role
        self.start = start
        self.end = end
        self.page = page

        self.content = content


class DocumentNodeList:
    """A helper class to aggregate parse results into non-overlapping chunks"""

    def __init__(self) -> None:
        self.docs: List[DocumentNode] = []

    def _does_overlap(self, new_doc: DocumentNode) -> bool:
        for doc in self.docs:
            if doc.start < new_doc.end and new_doc.start < doc.end:
                return True
        return False

    def add_doc_node(self, doc_node: DocumentNode) -> None:
        if not self._does_overlap(doc_node):
            self.docs.append(doc_node)
            self.docs.sort(key=lambda dn: dn.start)


class DocumentIntelligenceDetailParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence (formerly Forms Recognizer)
    and chunks at character level."""

    def __init__(self, client: Any, model: str, title: str = "") -> None:
        self.client = client
        self.model = model
        self.title = title

    def _combine_result(
        self,
        result: Any,
    ) -> Sequence[DocumentNode]:
        """Generate an order list of non-overlapping document nodes
        with content and some metadata"""

        doc_node_list = DocumentNodeList()

        # Tables first as they overlap with the other types
        # (text an kvp can appear in tables)
        for tbl in result.tables:
            start = tbl.spans[0].offset
            last_span = tbl.spans[-1]
            end = last_span.offset + last_span.length
            page = tbl.bounding_regions[0].page_number

            # generate table content
            content = ""

            dn = DocumentNode(
                "TABLE", "", start, end, content, page
            )  ### Table has no content attribute!
            doc_node_list.add_doc_node(dn)

        for kvp in result.key_value_pairs:
            if kvp.key and kvp.value:
                # Assume KVP is a continuous set of characters in the document (?)

                start = kvp.key.spans[0].offset
                end = kvp.value.spans[-1].offset + kvp.value.spans[-1].length
                page = kvp.key.bounding_regions[0].page_number

                dn = DocumentNode(
                    "KVP",
                    "",
                    start,
                    end,
                    kvp.key.content + " = " + kvp.value.content,
                    page,
                )
                doc_node_list.add_doc_node(dn)

        for p in result.paragraphs:
            start = p.spans[0].offset
            last_span = p.spans[-1]
            end = last_span.offset + last_span.length
            role = "paragraph" if (p.role is None) or (p.role == "") else p.role
            page = p.bounding_regions[0].page_number

            dn = DocumentNode("TEXT", role, start, end, p.content, page)
            doc_node_list.add_doc_node(dn)

        return doc_node_list.docs

    def _generate_docs(self, blob: Blob, result: Any) -> Sequence[Document]:
        """Combine nodes that appear under same section heading
        (just linear, no subsubheadings or whatsoever)"""
        current_heading = ""
        current_section_startpage = 1
        current_section_content = ""
        current_title = self.title

        doc_nodes = self._combine_result(result)

        docs = []

        for dn in doc_nodes:
            if dn.type == "TEXT" and dn.role == "paragraph":
                current_section_content += " " + dn.content + "\n"
            elif dn.type == "TEXT" and dn.role == "sectionHeading":
                if len(current_section_content) > 0:
                    d = Document(
                        page_content=current_section_content,
                        metadata={
                            "source": blob.source,
                            "page": current_section_startpage,
                            "section": current_heading,
                            "title": current_title,
                            "role": "paragraph",
                        },
                    )
                    docs.append(d)

                # Reset for new section
                current_heading = dn.content
                current_section_startpage = dn.page
                current_section_content = ""
            elif (
                (dn.type == "TEXT")
                and (dn.role == "title")
                and (len(current_title) == 0)
            ):
                current_title = dn.content
            elif dn.type == "KVP":
                d = Document(
                    page_content=dn.content,
                    metadata={
                        "source": blob.source,
                        "page": current_section_startpage,
                        "section": current_heading,
                        "title": current_title,
                        "role": "kvp",
                    },
                )
                docs.append(d)

        # add final paragraph
        if len(current_section_content) > 0:
            d = Document(
                page_content=current_section_content,
                metadata={
                    "source": blob.source,
                    "page": current_section_startpage,
                    "section": current_heading,
                    "title": current_title,
                    "role": "paragraph",
                },
            )
            docs.append(d)

        return docs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()

            docs = self._generate_docs(blob, result)

            yield from docs
