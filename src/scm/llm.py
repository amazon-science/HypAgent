# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Model-agnostic LLM interface.

A single ``generate(system, user, temperature, max_tokens) -> LLMResponse``
contract, implemented for Anthropic, OpenAI, and AWS Bedrock. Add a provider by
subclassing :class:`LLMProvider` and registering it in :func:`get_provider`.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict  # {"input_tokens": int, "output_tokens": int}


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, user: str,
                 temperature: float = 0.7, max_tokens: int = 4096) -> LLMResponse:
        ...


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model

    def generate(self, system, user, temperature=0.7, max_tokens=4096):
        r = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return LLMResponse(
            content=r.content[0].text, model=r.model,
            usage={"input_tokens": r.usage.input_tokens,
                   "output_tokens": r.usage.output_tokens},
        )


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model

    def generate(self, system, user, temperature=0.7, max_tokens=4096):
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        c = r.choices[0]
        return LLMResponse(
            content=c.message.content, model=r.model,
            usage={"input_tokens": r.usage.prompt_tokens,
                   "output_tokens": r.usage.completion_tokens},
        )


class BedrockProvider(LLMProvider):
    """AWS Bedrock (Anthropic-on-Bedrock message format).

    Requires ``boto3`` and valid AWS credentials. Configuration via environment:
        BEDROCK_MODEL_ID        model id (default: global.anthropic.claude-sonnet-4-6)
        AWS_REGION              region (default: us-east-1)
        AWS_PROFILE             profile name (default: default)
        CREDENTIAL_REFRESH_CMD  optional shell command run once on an ExpiredToken
                                error before retrying (e.g. "aws sso login").

    Retries transient errors with exponential backoff. Connect/read timeouts are
    bounded so a stalled socket surfaces as a retryable error rather than hanging.
    """

    def __init__(self, model=None, region=None, profile=None):
        import boto3
        self._boto3 = boto3
        self._profile = profile or os.environ.get("AWS_PROFILE", "default")
        self._region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.model = model or os.environ.get("BEDROCK_MODEL_ID",
                                             "global.anthropic.claude-sonnet-4-6")
        self._create_client()

    def _create_client(self):
        from botocore.config import Config
        creds_exist = (os.path.exists(os.path.expanduser("~/.aws/credentials")) or
                       os.path.exists(os.path.expanduser("~/.aws/config")))
        # A fresh Session re-reads credentials from disk (a reused one keeps the
        # first-loaded object and fails after an on-disk rotation).
        session = self._boto3.Session(profile_name=self._profile if creds_exist else None)
        cfg = Config(connect_timeout=int(os.environ.get("BEDROCK_CONNECT_TIMEOUT", "15")),
                     read_timeout=int(os.environ.get("BEDROCK_READ_TIMEOUT", "60")),
                     tcp_keepalive=True, retries={"max_attempts": 0})
        self.client = session.client("bedrock-runtime", region_name=self._region, config=cfg)

    def _refresh_credentials(self):
        cmd = os.environ.get("CREDENTIAL_REFRESH_CMD")
        if cmd:
            import subprocess
            try:
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True, timeout=120)
            except subprocess.TimeoutExpired:
                pass
        self._create_client()

    def generate(self, system, user, temperature=0.7, max_tokens=4096):
        import time
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens, "temperature": temperature,
            "system": system, "messages": [{"role": "user", "content": user}],
        })
        result = None
        for attempt in range(6):
            try:
                resp = self.client.invoke_model(modelId=self.model, body=body)
                result = json.loads(resp["body"].read())
                break
            except Exception as e:
                err = str(e)
                etype = type(e).__name__
                print(f"    [Bedrock attempt {attempt + 1}/6: {etype}: {err[:200]}]", flush=True)
                if "AccessDeniedException" in etype or (
                        "ValidationException" in etype and "malformed" in err.lower()):
                    raise  # not transient
                if "ExpiredToken" in err or "expired" in err.lower():
                    self._refresh_credentials()
                    time.sleep(min(5 * (2 ** attempt), 60))
                    continue
                if any(k in err or k in etype for k in
                       ("Connection", "Timeout", "BrokenPipe", "nodename")):
                    self._create_client()
                time.sleep(min(15 * (2 ** attempt), 300))
        else:
            raise RuntimeError("Bedrock failed after 6 retries")
        return LLMResponse(content=result["content"][0]["text"],
                           model=self.model, usage=result.get("usage", {}))


_PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "bedrock": BedrockProvider,
}


def get_provider(provider: str = "anthropic", **kwargs) -> LLMProvider:
    if provider not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](**kwargs)
