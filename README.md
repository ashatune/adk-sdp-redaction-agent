# ADK + Cloud SDP Resilient Redaction Agent

Reference GenAI enterprise assistant built with **Google's Agent Development Kit (ADK)** in Python, combining **Cloud Model Armor** pre-inference prompt screening, **Vertex AI Search (Discovery Engine)** document grounding, and **Google Cloud Sensitive Data Protection (SDP / DLP v2)** post-inference in-place redaction.

Instead of relying on binary Allow/Block output guardrails that reject grounded responses containing incidental PII from enterprise documents, this agent:
1. **Screens & sanitizes incoming prompts** via `before_model_callback` using **Cloud Model Armor** (`legal-prompt-template` binary block).
2. **Generates grounded responses** using **Gemini 2.5** (`gemini-2.5-flash`) on Vertex AI connected to an existing **Gemini Enterprise / Discovery Engine Data Store**.
3. **Intercepts & redacts sensitive entities in-place** post-inference via `after_model_callback` calling `DlpServiceClient.deidentify_content()` (`[US_SOCIAL_SECURITY_NUMBER]`, `[EMAIL_ADDRESS]`, `[CREDIT_CARD_NUMBER]`, `[GCP_API_KEY]`) while preserving surrounding business context.

For the full architecture diagram, regional/multi-region alignment, least-privilege IAM setup, and Gemini Enterprise registration walkthrough, see **[docs/CUSTOMER_IMPLEMENTATION_GUIDE.md](docs/CUSTOMER_IMPLEMENTATION_GUIDE.md)**.

## Configuration

Copy `.env.example` or export your target Google Cloud resource parameters:

| Environment Variable | Default Placeholder | Description |
| :--- | :--- | :--- |
| `GCP_PROJECT_ID` | `your-gcp-project-id` | Target Google Cloud project ID |
| `GCP_PROJECT_NUMBER` | `000000000000` | Target Google Cloud project number |
| `SDP_LOCATION` | `us` | Multi-region/region for Cloud SDP & Model Armor templates |
| `AGENT_ENGINE_REGION` | `us-west1` | Regional compute location for Vertex AI Agent Engine |
| `VERTEX_LOCATION` | `global` | Vertex AI Gemini & Discovery Engine location |
| `SDP_INSPECT_TEMPLATE_PATH` | `projects/.../inspectTemplates/legal-sdp-inspect` | Cloud SDP InspectTemplate resource name |
| `SDP_DEIDENTIFY_TEMPLATE_PATH` | `projects/.../deidentifyTemplates/legal-deid-template` | Cloud SDP DeidentifyTemplate resource name |
| `MODEL_ARMOR_PROMPT_TEMPLATE_ID` | `legal-prompt-template` | Model Armor binary prompt filter template ID |
| `GEMINI_ENTERPRISE_ENGINE_ID` | `your-gemini-enterprise-engine-id` | Gemini Enterprise app engine ID |
| `DISCOVERY_ENGINE_DATA_STORE_ID` | `projects/.../dataStores/your-data-store-id` | Connected Discovery Engine Data Store resource path |

## Quickstart

### 1. Install Dependencies (`uv` or `pip`)

```bash
# Using uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using standard pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Offline Verification Suite

```bash
pytest -v
```

### 3. Run Standalone CLI Evaluator

```bash
# Deterministic offline simulation suite (no GCP credentials required)
python run_eval.py --mode simulate --suite

# Live GCP Vertex AI + Cloud SDP single-shot test (requires ADC + GCP_PROJECT_ID)
export GCP_PROJECT_ID="your-gcp-project-id"
python run_eval.py --mode live --prompt "Draft an intake summary using mock SSN 000-00-0000 and test@example.com."

# Interactive REPL
python run_eval.py --mode live
```

### 4. Start FastAPI `/chat` Server

```bash
uvicorn app.server:app --reload --port 8080
```

Test with `curl`:

```bash
curl -s http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Summarize clause 4.2 mutual indemnification terms."}' | jq
```

### 5. Deploy to Vertex AI Agent Engine & Register in Gemini Enterprise

```bash
python deploy_agent.py \
  --project-id "${GCP_PROJECT_ID}" \
  --project-number "${GCP_PROJECT_NUMBER}" \
  --region "us-west1" \
  --ge-engine-id "${GEMINI_ENTERPRISE_ENGINE_ID}"
```
