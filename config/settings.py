"""Target environment and Cloud SDP / Model Armor / Data Store configuration."""

from __future__ import annotations

import os
from pydantic import BaseModel, Field

# Default Environment & Resource Placeholders (override via environment variables)
PROJECT_ID = "your-gcp-project-id"
PROJECT_NUMBER = "000000000000"
LOCATION = "us"
AGENT_ENGINE_REGION = "us-west1"

INSPECT_TEMPLATE_ID = "legal-sdp-inspect"
FALLBACK_INSPECT_TEMPLATE_ID = "legal-stp-inspect"
DEIDENTIFY_TEMPLATE_ID = "legal-deid-template"

INSPECT_TEMPLATE_PATH = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/inspectTemplates/{INSPECT_TEMPLATE_ID}"
)
FALLBACK_INSPECT_TEMPLATE_PATH = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/inspectTemplates/{FALLBACK_INSPECT_TEMPLATE_ID}"
)
DEIDENTIFY_TEMPLATE_PATH = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/deidentifyTemplates/{DEIDENTIFY_TEMPLATE_ID}"
)

# Model Armor Template Identifiers & Resource Paths
PROMPT_FILTER_TEMPLATE_ID = "legal-prompt-template"
RESPONSE_FILTER_TEMPLATE_ID = "legal-response-template"
PROMPT_FILTER_TEMPLATE_PATH = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{PROMPT_FILTER_TEMPLATE_ID}"
)
RESPONSE_FILTER_TEMPLATE_PATH = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{RESPONSE_FILTER_TEMPLATE_ID}"
)

# Gemini Enterprise App & Connected GCS Data Store (global)
GEMINI_ENTERPRISE_ENGINE_ID = "your-gemini-enterprise-engine-id"
DATA_STORE_ID = (
    f"projects/{PROJECT_NUMBER}/locations/global/collections/default_collection/"
    "dataStores/your-data-store-id"
)

# Vertex AI / Gemini Inference Defaults (pinned to global regardless of Agent Engine region)
VERTEX_LOCATION = "global"
GEMINI_MODEL = "gemini-2.5-flash"
FALLBACK_GEMINI_MODEL = "gemini-3.5-flash"


class AgentSettings(BaseModel):
    """Typed runtime settings initialized with environment-configurable defaults."""

    project_id: str = Field(default=PROJECT_ID)
    project_number: str = Field(default=PROJECT_NUMBER)
    location: str = Field(default=LOCATION)
    agent_engine_region: str = Field(default=AGENT_ENGINE_REGION)
    vertex_location: str = Field(default=VERTEX_LOCATION)
    gemini_model: str = Field(default=GEMINI_MODEL)
    fallback_gemini_model: str = Field(default=FALLBACK_GEMINI_MODEL)
    inspect_template_path: str = Field(default=INSPECT_TEMPLATE_PATH)
    fallback_inspect_template_path: str = Field(default=FALLBACK_INSPECT_TEMPLATE_PATH)
    deidentify_template_path: str = Field(default=DEIDENTIFY_TEMPLATE_PATH)
    prompt_filter_template_id: str = Field(default=PROMPT_FILTER_TEMPLATE_ID)
    response_filter_template_id: str = Field(default=RESPONSE_FILTER_TEMPLATE_ID)
    prompt_filter_template_path: str = Field(default=PROMPT_FILTER_TEMPLATE_PATH)
    response_filter_template_path: str = Field(default=RESPONSE_FILTER_TEMPLATE_PATH)
    gemini_enterprise_engine_id: str = Field(default=GEMINI_ENTERPRISE_ENGINE_ID)
    data_store_id: str = Field(default=DATA_STORE_ID)
    enable_datastore_search: bool = Field(default=True)
    enable_model_armor_api: bool = Field(
        default=True,
        description=(
            "When True, calls the live Cloud Model Armor API in addition to local "
            "deterministic prompt-injection screening."
        ),
    )

    @property
    def sdp_parent(self) -> str:
        """Parent resource path for regional/multi-regional Cloud SDP calls."""
        return f"projects/{self.project_id}/locations/{self.location}"


def get_settings() -> AgentSettings:
    """Load AgentSettings with environment variable overrides."""
    project_id = os.getenv("GCP_PROJECT_ID", PROJECT_ID)
    project_number = os.getenv("GCP_PROJECT_NUMBER", PROJECT_NUMBER)
    location = os.getenv("SDP_LOCATION", LOCATION)
    prompt_template_id = os.getenv(
        "MODEL_ARMOR_PROMPT_TEMPLATE_ID", PROMPT_FILTER_TEMPLATE_ID
    )
    response_template_id = os.getenv(
        "MODEL_ARMOR_RESPONSE_TEMPLATE_ID", RESPONSE_FILTER_TEMPLATE_ID
    )

    return AgentSettings(
        project_id=project_id,
        project_number=project_number,
        location=location,
        agent_engine_region=os.getenv("AGENT_ENGINE_REGION", AGENT_ENGINE_REGION),
        vertex_location=os.getenv("VERTEX_LOCATION", VERTEX_LOCATION),
        gemini_model=os.getenv("GEMINI_MODEL", GEMINI_MODEL),
        fallback_gemini_model=os.getenv(
            "FALLBACK_GEMINI_MODEL", FALLBACK_GEMINI_MODEL
        ),
        inspect_template_path=os.getenv(
            "SDP_INSPECT_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/inspectTemplates/{INSPECT_TEMPLATE_ID}",
        ),
        fallback_inspect_template_path=os.getenv(
            "SDP_FALLBACK_INSPECT_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/inspectTemplates/{FALLBACK_INSPECT_TEMPLATE_ID}",
        ),
        deidentify_template_path=os.getenv(
            "SDP_DEIDENTIFY_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/deidentifyTemplates/{DEIDENTIFY_TEMPLATE_ID}",
        ),
        prompt_filter_template_id=prompt_template_id,
        response_filter_template_id=response_template_id,
        prompt_filter_template_path=os.getenv(
            "MODEL_ARMOR_PROMPT_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/templates/{prompt_template_id}",
        ),
        response_filter_template_path=os.getenv(
            "MODEL_ARMOR_RESPONSE_TEMPLATE_PATH",
            f"projects/{project_id}/locations/{location}/templates/{response_template_id}",
        ),
        gemini_enterprise_engine_id=os.getenv(
            "GEMINI_ENTERPRISE_ENGINE_ID", GEMINI_ENTERPRISE_ENGINE_ID
        ),
        data_store_id=os.getenv(
            "DISCOVERY_ENGINE_DATA_STORE_ID",
            f"projects/{project_number}/locations/global/collections/default_collection/dataStores/your-data-store-id",
        ),
        enable_datastore_search=os.getenv("ENABLE_DATASTORE_SEARCH", "true").lower()
        in {"1", "true", "yes"},
        enable_model_armor_api=os.getenv("ENABLE_MODEL_ARMOR_API", "true").lower()
        in {"1", "true", "yes"},
    )
