import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI(title="Preprocessing Service")


class TextRequest(BaseModel):
    text: str


class DocRequest(BaseModel):
    doc: Dict
    dataset_name: str


class PreprocessingService:

    def __init__(self):

        print("[*] Loading SpaCy...")

        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner"]
        )

    def clean_text(
        self,
        text: str
    ) -> List[str]:

        if (
            not text
            or not isinstance(text, str)
            or not text.strip()
        ):
            return []

        doc = self.nlp(
            text.lower()
        )

        return [

            token.lemma_

            for token in doc

            if (
                not token.is_stop
                and not token.is_punct
                and token.text.strip()
            )

        ]

    def extract_and_clean(
        self,
        doc_dict: dict,
        dataset_name: str
    ) -> List[str]:

        if not doc_dict:
            return []

        dataset = (
            dataset_name
            .lower()
        )

        raw_text = ""

        # ======================
        # CORD19
        # ======================

        if (
            "cord19"
            in dataset
            or
            "trec-covid"
            in dataset
        ):

            title = (
                doc_dict.get(
                    "title",
                    ""
                )
                or ""
            )

            abstract = (
                doc_dict.get(
                    "abstract",
                    ""
                )
                or ""
            )

            raw_text = (
                title
                + " "
                + abstract
            )

        # ======================
        # CLINICAL
        # ======================

        elif (
            "clinical"
            in dataset
        ):

            title = (
                doc_dict.get(
                    "title",
                    ""
                )
                or ""
            )

            summary = (
                doc_dict.get(
                    "summary",
                    ""
                )
                or ""
            )

            detailed = (
                doc_dict.get(
                    "detailed_description",
                    ""
                )
                or ""
            )

            eligibility = (
                doc_dict.get(
                    "eligibility",
                    ""
                )
                or ""
            )

            raw_text = " ".join([

                title,

                summary,

                detailed,

                eligibility

            ])

        # ======================
        # DEFAULT
        # ======================

        else:

            raw_text = (

                doc_dict.get(
                    "text",
                    ""
                )

                or

                doc_dict.get(
                    "body",
                    ""
                )

            )

        return self.clean_text(
            raw_text
        )


preprocessor_logic = (
    PreprocessingService()
)


@app.post("/clean-text")
async def clean_text_endpoint(
    request: TextRequest
):

    try:

        tokens = (
            preprocessor_logic
            .clean_text(
                request.text
            )
        )

        return {

            "status":
            "success",

            "tokens":
            tokens

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/extract-and-clean")
async def extract_and_clean_endpoint(
    request: DocRequest
):

    try:

        tokens = (

            preprocessor_logic
            .extract_and_clean(

                request.doc,

                request.dataset_name

            )

        )

        return {

            "status":
            "success",

            "tokens":
            tokens

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/health")
async def health():

    return {
        "status":
        "running"
    }


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "preprocessing_service:app",
        host="127.0.0.1",
        port=8001,
        reload=False
    )