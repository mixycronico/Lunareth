# plugins/codex/processors/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class CmdBase(BaseModel):
    action: str

class PluginParams(BaseModel):
    plugin_name: str = Field(..., regex=r"^[a-zA-Z0-9_-]+$")

class WebsiteParams(BaseModel):
    template: Literal["react","fastapi"]
    project_name: str = Field(..., regex=r"^[a-zA-Z0-9_-]+$")

class ReviseParams(BaseModel):
    file: str

class CmdGeneratePlugin(CmdBase):
    action: Literal["generate_plugin"]
    params: PluginParams

class CmdGenerateWebsite(CmdBase):
    action: Literal["generate_website"]
    params: WebsiteParams

class CmdRevise(CmdBase):
    action: Literal["revise"]
    params: ReviseParams

# Unión de todos los comandos permitidos
from typing import Union
Cmd = Union[CmdGeneratePlugin, CmdGenerateWebsite, CmdRevise]