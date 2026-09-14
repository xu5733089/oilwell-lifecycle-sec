"""HTTP 接口层（方案 §9.2）。

用 Starlette 而不是 FastAPI：FastAPI 就建在 Starlette 之上，
本层只做「解析参数 -> 调 services -> 返回 JSON」，不含任何业务逻辑，
所以换成 FastAPI 只需把下面的 `route` 函数改成带类型注解的 `@app.post`，
services.py 一行都不用动。内网离线环境少一个依赖就少一份风险。

所有响应都带 model_version / label_def_version / data_source / trace_id 四个追溯字段
—— 这是可审计性的技术实现，由 services 层统一注入。
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict

from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .. import trace
from ..agent.orchestrator import Agent
from ..agent.tools import schemas
from . import services as S

API = "/api/v1"


def _ok(data: Dict) -> JSONResponse:
    return JSONResponse(data)


def _err(msg: str, code: int = 400, trace_id: str | None = None) -> JSONResponse:
    return JSONResponse({"error": msg, "trace_id": trace_id or trace.new_trace_id("err")},
                        status_code=code)


# GET 的查询参数一律是字符串。只对**已知的数值参数**做转换，
# 井号这类绝不猜类型 —— "0001" 猜成整数会把井号毁掉。
NUMERIC_PARAMS = {"limit": int, "top_k": int, "mc_samples": int, "obs_days": int,
                  "n_boot": int, "d_min_year": float}


def _coerce(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    for k, cast in NUMERIC_PARAMS.items():
        if k in out and isinstance(out[k], str):
            try:
                out[k] = cast(out[k])
            except ValueError:
                pass          # 转不了就原样传下去，让 services 报参数不合法
    return out


async def _body(request: Request) -> Dict[str, Any]:
    if request.method == "GET":
        return _coerce(dict(request.query_params))
    try:
        return await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}


def _wrap(fn: Callable[..., Dict], *names: str):
    """把 services 函数包成 HTTP handler：缺必填参数返回 400，内核异常返回 422。"""
    async def handler(request: Request) -> JSONResponse:
        body = await _body(request)
        missing = [n for n in names if n not in body]
        if missing:
            return _err(f"缺少必填参数：{missing}", 400)
        try:
            return _ok(fn(**body))
        except S.KernelError as exc:
            return _err(str(exc), 422)
        except TypeError as exc:
            return _err(f"参数不合法：{exc}", 400)
    return handler


async def health(request: Request) -> JSONResponse:
    try:
        meta = S._bundle()["meta"]
        return _ok(dict(status="ok", model_version=meta["model_version"],
                        label_def_version=meta["label_def_version"],
                        data_source=meta["data_source"], backend=meta["backend"]))
    except Exception as exc:
        return _err(f"模型未就绪：{exc}", 503)


async def tools_schema(request: Request) -> JSONResponse:
    """把工具清单以 OpenAI function-calling schema 暴露出去，
    便于后续在油博士平台上以 skill 形式注册同一套工具。"""
    return _ok({"tools": schemas()})


async def ask(request: Request) -> JSONResponse:
    body = await _body(request)
    q = body.get("question")
    if not q:
        return _err("缺少必填参数：question", 400)
    return _ok(Agent().answer(str(q)).to_dict())


WEB_DIR = Path(__file__).resolve().parents[2] / "web"


async def index(request: Request):
    """前端是零依赖单文件：不需要 node、不需要 npm、不连 CDN。

    内网装不了 npm 也连不了外网，Vue 全家桶会在部署那天卡住；
    图表全部手写 SVG，页面直接由本服务托管。
    """
    f = WEB_DIR / "index.html"
    if not f.exists():
        return _err("前端文件缺失：web/index.html", 404)
    return FileResponse(f)


routes = [
    Route("/", index, methods=["GET"]),
    Route("/health", health, methods=["GET"]),
    Route(f"{API}/wells", _wrap(S.list_wells), methods=["POST", "GET"]),
    Route(f"{API}/well/curve", _wrap(S.well_curve, "well_code"), methods=["POST", "GET"]),
    Route(f"{API}/overview", _wrap(S.overview), methods=["POST", "GET"]),
    Route(f"{API}/eval/summary", _wrap(S.eval_summary), methods=["POST", "GET"]),
    Route(f"{API}/tools", tools_schema, methods=["GET"]),
    Route(f"{API}/agent/ask", ask, methods=["POST"]),
    Route(f"{API}/well/query", _wrap(S.query_well, "well_code"), methods=["POST", "GET"]),
    Route(f"{API}/predict/lifecycle", _wrap(S.predict_lifecycle, "well_code"),
          methods=["POST", "GET"]),
    Route(f"{API}/analogs", _wrap(S.find_analog_wells, "well_code"), methods=["POST", "GET"]),
    Route(f"{API}/reserves/dca", _wrap(S.fit_dca, "well_code"), methods=["POST", "GET"]),
    Route(f"{API}/reserves/volumetric", _wrap(S.estimate_reserves_volumetric, "well_code"),
          methods=["POST", "GET"]),
    Route(f"{API}/reserves/crosscheck", _wrap(S.cross_check_reserves, "well_code"),
          methods=["POST", "GET"]),
    Route(f"{API}/sec/screen", _wrap(S.sec_screen, "well_code"), methods=["POST", "GET"]),
    Route(f"{API}/sec/reconcile", _wrap(S.reserves_reconcile, "values"), methods=["POST"]),
]

if WEB_DIR.exists():
    routes.append(Mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static"))

app = Starlette(routes=routes)
