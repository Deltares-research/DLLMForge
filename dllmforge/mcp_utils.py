"""
Generic MCP server plumbing for DLLMForge (no database knowledge). Requires the ``mcp`` extra.

- :func:`create_server` / :func:`run`: server creation and transport selection from ``<PREFIX>_*`` env vars.
- :func:`auth_from_env` / :class:`EntraTokenVerifier`: optional Microsoft Entra ID bearer-token auth.
- :class:`InstructionAck` / :func:`with_instruction_ack`: optionally require clients to load the
  server instructions before calling selected tools.
- :func:`register_functions`: expose plain typed Python functions as MCP tools, by function name.

Nothing here starts an LLM or agent workflow: MCP clients run their own.
"""
import asyncio
import functools
import hashlib
import inspect
import os
from typing import Any, Callable, Iterable, Optional, Tuple

from typing_extensions import TypedDict

from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

__all__ = [
    "EntraTokenVerifier", "auth_from_env", "env_flag", "InstructionsPayload", "InstructionAck", "with_instruction_ack",
    "register_functions", "create_server", "run"
]

_TRUE = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in _TRUE


class EntraTokenVerifier:
    """Validate Microsoft Entra ID v2 access tokens (RS256, issuer, audience, expiry)."""

    def __init__(self, tenant_id: str, audience: str) -> None:
        import jwt  # pyjwt, part of the ``mcp`` extra
        self._jwt = jwt
        self.issuer = f"https://login.microsoftonline.com/{tenant_id}/v2.0"
        self.audience = audience
        self.jwks_client = jwt.PyJWKClient(f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys")

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        jwt = self._jwt
        try:
            signing_key = await asyncio.to_thread(self.jwks_client.get_signing_key_from_jwt, token)
            claims = jwt.decode(token,
                                signing_key.key,
                                algorithms=["RS256"],
                                audience=self.audience,
                                issuer=self.issuer,
                                options={"require": ["exp", "iss", "sub"]})
        except jwt.PyJWTError:
            return None
        return AccessToken(token=token,
                           client_id=str(claims.get("azp") or claims.get("appid") or "unknown"),
                           scopes=str(claims.get("scp", "")).split(),
                           expires_at=int(claims["exp"]),
                           subject=str(claims["sub"]),
                           claims=claims)


def auth_from_env(prefix: str) -> Optional[Tuple[EntraTokenVerifier, AuthSettings]]:
    """Entra auth from ``<PREFIX>_ENTRA_TENANT_ID``, ``_ENTRA_AUDIENCE``, ``_MCP_RESOURCE_URL`` and
    optional ``_MCP_REQUIRED_SCOPES``. Returns ``None`` (no auth) unless the first three are set."""
    tenant_id = os.getenv(f"{prefix}_ENTRA_TENANT_ID")
    audience = os.getenv(f"{prefix}_ENTRA_AUDIENCE")
    resource_url = os.getenv(f"{prefix}_MCP_RESOURCE_URL")
    if not (tenant_id and audience and resource_url):
        return None
    settings = AuthSettings(issuer_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
                            resource_server_url=resource_url,
                            required_scopes=os.getenv(f"{prefix}_MCP_REQUIRED_SCOPES", "").split() or None)
    return EntraTokenVerifier(tenant_id, audience), settings


class InstructionsPayload(TypedDict):
    instructions_version: str
    system_prompt: str
    usage: str


class InstructionAck:
    """Versioned server instructions that clients may be required to acknowledge.

    Args:
        instructions: Callable returning the current instructions text.
        version_prefix: Prefix of the version string, e.g. ``"chw-instructions"``.
        required: Whether tools wrapped with :func:`with_instruction_ack` enforce the version.
        instructions_tool: Name of the tool that returns the payload (used in error messages).
    """

    def __init__(self,
                 instructions: Callable[[], str],
                 version_prefix: str,
                 required: bool = False,
                 instructions_tool: str = "get_instructions"):
        self._instructions = instructions
        self.version_prefix = version_prefix
        self.required = required
        self.instructions_tool = instructions_tool

    def version(self, text: Optional[str] = None) -> str:
        text = self._instructions() if text is None else text
        return f"{self.version_prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"

    def payload(self) -> InstructionsPayload:
        text = self._instructions()
        usage = "Load these instructions into your agent instructions before answering questions."
        if self.required:
            usage += " Pass instructions_version to the query tools."
        return InstructionsPayload(instructions_version=self.version(text), system_prompt=text, usage=usage)

    def check(self, instructions_version: Optional[str]) -> None:
        if not self.required:
            return
        expected = self.version()
        if instructions_version != expected:
            # ToolError (not ValueError): the SDK only forwards ToolError messages to clients.
            raise ToolError(f"Instructions have not been loaded or are stale. Call {self.instructions_tool} first "
                            f"and retry with instructions_version='{expected}'.")


def with_instruction_ack(func: Callable[..., Any], ack: InstructionAck) -> Callable[..., Any]:
    """Return ``func`` with an extra optional ``instructions_version`` argument checked by ``ack``."""

    @functools.wraps(func)
    def wrapper(*args: Any, instructions_version: Optional[str] = None, **kwargs: Any) -> Any:
        ack.check(instructions_version)
        return func(*args, **kwargs)

    signature = inspect.signature(func)
    extra = inspect.Parameter("instructions_version",
                              inspect.Parameter.KEYWORD_ONLY,
                              default=None,
                              annotation=Optional[str])
    wrapper.__signature__ = signature.replace(parameters=[*signature.parameters.values(), extra])
    wrapper.__doc__ = (func.__doc__
                       or "").rstrip() + ("\n\nIf instruction acknowledgement is enabled, pass instructions_version "
                                          f"from {ack.instructions_tool}.")
    return wrapper


def register_functions(server: MCPServer, functions: Iterable[Callable[..., Any]], read_only: bool = True) -> None:
    """Register plain functions as MCP tools under their ``__name__`` (docstring = description)."""
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False) if read_only else None
    for func in functions:
        server.add_tool(func,
                        name=func.__name__,
                        description=inspect.cleandoc(func.__doc__ or func.__name__),
                        annotations=annotations)


def create_server(name: str, env_prefix: str, instructions: Optional[str] = None) -> MCPServer:
    """Create an :class:`MCPServer`, with Entra auth when configured via ``<PREFIX>_*`` env vars."""
    auth = auth_from_env(env_prefix)
    return MCPServer(name,
                     instructions=instructions,
                     token_verifier=auth[0] if auth else None,
                     auth=auth[1] if auth else None)


def run(server: MCPServer, env_prefix: str) -> None:
    """Run ``server`` using ``<PREFIX>_MCP_TRANSPORT`` (stdio | streamable-http | sse), ``_MCP_HOST``,
    ``_MCP_PORT`` and ``_MCP_PATH``."""
    transport = os.getenv(f"{env_prefix}_MCP_TRANSPORT", "stdio")
    host = os.getenv(f"{env_prefix}_MCP_HOST", "127.0.0.1")
    port = int(os.getenv(f"{env_prefix}_MCP_PORT", "8000"))
    if transport == "streamable-http":
        server.run(transport="streamable-http",
                   host=host,
                   port=port,
                   streamable_http_path=os.getenv(f"{env_prefix}_MCP_PATH", "/mcp"))
    elif transport == "sse":
        server.run(transport="sse", host=host, port=port)
    elif transport == "stdio":
        server.run(transport="stdio")
    else:
        raise ValueError(f"Unknown {env_prefix}_MCP_TRANSPORT {transport!r}; use stdio, streamable-http or sse")
