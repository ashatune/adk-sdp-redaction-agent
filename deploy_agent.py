#!/usr/bin/env python3
"""Deploy ADK SDP Redaction Agent to Vertex AI Agent Engine & register in Gemini Enterprise."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import requests

from config.settings import get_settings

REPO_ROOT = Path(__file__).resolve().parent
DISPLAY_NAME = "Enterprise SDP Redaction Agent"
DESCRIPTION = (
    "High-code ADK assistant grounded on Discovery Engine with "
    "Model Armor prompt screening and post-inference Cloud SDP in-place redaction."
)


def get_access_token() -> str:
    return (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode("utf-8")
        .strip()
    )


def find_existing_reasoning_engine(
    project_id: str,
    project_number: str,
    region: str,
    display_name: str,
) -> str | None:
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "X-Goog-User-Project": project_id,
    }
    url = (
        f"https://{region}-aiplatform.googleapis.com/v1beta1/"
        f"projects/{project_number}/locations/{region}/reasoningEngines"
    )
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    for engine in resp.json().get("reasoningEngines", []):
        if engine.get("displayName") == display_name:
            return engine["name"]
    return None


def deploy_reasoning_engine(
    project_id: str,
    project_number: str,
    region: str,
    service_account: str,
) -> str:
    existing_name = find_existing_reasoning_engine(
        project_id, project_number, region, DISPLAY_NAME
    )
    venv_adk = REPO_ROOT / ".venv" / "bin" / "adk"
    adk_bin = str(venv_adk) if venv_adk.exists() else (shutil.which("adk") or "adk")
    cmd = [
        adk_bin,
        "deploy",
        "agent_engine",
        f"--project={project_id}",
        f"--region={region}",
        f"--display_name={DISPLAY_NAME}",
        f"--description={DESCRIPTION}",
        f"--service_account={service_account}",
        f"--extra_packages={REPO_ROOT / 'config'}",
    ]
    if existing_name:
        engine_id = existing_name.split("/")[-1]
        print(f"Updating existing Reasoning Engine: {existing_name}")
        cmd.append(f"--agent_engine_id={engine_id}")
    else:
        print(f"Creating new Reasoning Engine in {region}...")

    cmd.append(str(REPO_ROOT / "app"))
    subprocess.run(cmd, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL, check=True)

    deployed_name = find_existing_reasoning_engine(
        project_id, project_number, region, DISPLAY_NAME
    )
    if not deployed_name:
        raise RuntimeError(
            "Reasoning Engine deployed, but resource lookup returned None."
        )
    print(f"Deployed Reasoning Engine: {deployed_name}")
    return deployed_name


def register_in_gemini_enterprise(
    project_id: str,
    project_number: str,
    ge_engine_id: str,
    reasoning_engine_name: str,
) -> dict:
    headers = {
        "Authorization": f"Bearer {get_access_token()}",
        "X-Goog-User-Project": project_id,
        "Content-Type": "application/json",
    }
    agents_url = (
        "https://discoveryengine.googleapis.com/v1alpha/"
        f"projects/{project_number}/locations/global/collections/default_collection/"
        f"engines/{ge_engine_id}/assistants/default_assistant/agents"
    )
    existing_resp = requests.get(agents_url, headers=headers, timeout=30)
    existing_resp.raise_for_status()
    for agent in existing_resp.json().get("agents", []):
        if agent.get("displayName") == DISPLAY_NAME:
            print(f"Agent already registered in Gemini Enterprise: {agent['name']}")
            patch_url = (
                f"https://discoveryengine.googleapis.com/v1alpha/{agent['name']}"
                "?updateMask=description,adkAgentDefinition"
            )
            payload = {
                "displayName": DISPLAY_NAME,
                "description": DESCRIPTION,
                "adkAgentDefinition": {
                    "toolSettings": {
                        "toolDescription": DESCRIPTION,
                    },
                    "provisionedReasoningEngine": {
                        "reasoningEngine": reasoning_engine_name,
                    },
                },
            }
            upd = requests.patch(patch_url, headers=headers, json=payload, timeout=30)
            upd.raise_for_status()
            return upd.json()

    payload = {
        "displayName": DISPLAY_NAME,
        "description": DESCRIPTION,
        "adkAgentDefinition": {
            "toolSettings": {
                "toolDescription": DESCRIPTION,
            },
            "provisionedReasoningEngine": {
                "reasoningEngine": reasoning_engine_name,
            },
        },
    }
    create_resp = requests.post(agents_url, headers=headers, json=payload, timeout=30)
    create_resp.raise_for_status()
    created = create_resp.json()
    print(f"Registered in Gemini Enterprise: {created.get('name')}")
    return created


def main() -> int:
    settings = get_settings()
    default_sa = f"adk-sdp-agent-sa@{settings.project_id}.iam.gserviceaccount.com"

    parser = argparse.ArgumentParser(
        description="Deploy ADK SDP Redaction Agent to Vertex AI Agent Engine & Gemini Enterprise."
    )
    parser.add_argument("--project-id", default=settings.project_id)
    parser.add_argument("--project-number", default=settings.project_number)
    parser.add_argument("--region", default=settings.agent_engine_region)
    parser.add_argument("--service-account", default=default_sa)
    parser.add_argument(
        "--ge-engine-id", default=settings.gemini_enterprise_engine_id
    )
    parser.add_argument("--skip-ge-register", action="store_true")
    args = parser.parse_args()

    re_name = deploy_reasoning_engine(
        project_id=args.project_id,
        project_number=args.project_number,
        region=args.region,
        service_account=args.service_account,
    )
    if not args.skip_ge_register:
        ge_agent = register_in_gemini_enterprise(
            project_id=args.project_id,
            project_number=args.project_number,
            ge_engine_id=args.ge_engine_id,
            reasoning_engine_name=re_name,
        )
        print(json.dumps(ge_agent, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
